#![allow(dead_code)]

use parking_lot::Mutex;
use std::sync::Arc;

/// Thread-safe circular buffer for audio samples
/// Used to pass audio between the producer thread and preprocessor thread
#[derive(Clone)]
pub struct RingBuffer {
    buffer: Arc<Mutex<CircularBuffer>>,
}

struct CircularBuffer {
    data: Vec<f32>,
    write_pos: usize,
    read_pos: usize,
    capacity: usize,
}

impl RingBuffer {
    /// Create a new ring buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(CircularBuffer {
                data: vec![0.0; capacity],
                write_pos: 0,
                read_pos: 0,
                capacity,
            })),
        }
    }

    /// Write samples to the buffer. Overwrites old data if buffer is full.
    /// Returns the number of samples actually written.
    pub fn write(&self, samples: &[f32]) -> usize {
        let mut buf = self.buffer.lock();
        let mut written = 0;

        for &sample in samples {
            let write_idx = buf.write_pos;
            buf.data[write_idx] = sample;
            buf.write_pos = (buf.write_pos + 1) % buf.capacity;
            written += 1;

            // If we've caught up to read position, advance read position
            if buf.write_pos == buf.read_pos && written > 1 {
                buf.read_pos = (buf.read_pos + 1) % buf.capacity;
            }
        }

        written
    }

    /// Read up to `len` samples from the buffer
    /// Returns the actual number of samples read
    pub fn read(&self, output: &mut [f32]) -> usize {
        let mut buf = self.buffer.lock();
        let mut read_count = 0;

        while read_count < output.len() && buf.read_pos != buf.write_pos {
            output[read_count] = buf.data[buf.read_pos];
            buf.read_pos = (buf.read_pos + 1) % buf.capacity;
            read_count += 1;
        }

        read_count
    }

    /// Get available samples to read without consuming them
    pub fn available(&self) -> usize {
        let buf = self.buffer.lock();
        if buf.write_pos >= buf.read_pos {
            buf.write_pos - buf.read_pos
        } else {
            buf.capacity - buf.read_pos + buf.write_pos
        }
    }

    /// Peek at samples without consuming them
    pub fn peek(&self, output: &mut [f32]) -> usize {
        let buf = self.buffer.lock();
        let mut read_count = 0;
        let mut pos = buf.read_pos;

        while read_count < output.len() && pos != buf.write_pos {
            output[read_count] = buf.data[pos];
            pos = (pos + 1) % buf.capacity;
            read_count += 1;
        }

        read_count
    }

    /// Clear all data in the buffer
    pub fn clear(&self) {
        let mut buf = self.buffer.lock();
        buf.read_pos = buf.write_pos;
    }

    /// Get the capacity of the buffer
    pub fn capacity(&self) -> usize {
        self.buffer.lock().capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_read() {
        let rb = RingBuffer::new(100);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        rb.write(&data);

        let mut output = vec![0.0; 5];
        let read_count = rb.read(&mut output);
        assert_eq!(read_count, 5);
        assert_eq!(output, data);
    }

    #[test]
    fn test_available() {
        let rb = RingBuffer::new(100);
        let data = vec![1.0, 2.0, 3.0];
        rb.write(&data);
        assert_eq!(rb.available(), 3);
    }

    #[test]
    fn test_wraparound() {
        let rb = RingBuffer::new(5);
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        rb.write(&data1);

        let mut output = vec![0.0; 2];
        rb.read(&mut output);
        assert_eq!(output, vec![1.0, 2.0]);

        let data2 = vec![6.0, 7.0];
        rb.write(&data2);

        let mut output = vec![0.0; 5];
        let read_count = rb.read(&mut output);
        assert_eq!(read_count, 5);
        assert_eq!(&output[..read_count], &[3.0, 4.0, 5.0, 6.0, 7.0]);
    }
}
