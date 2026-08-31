// Microphone capture for the live voice loop: downsample the context rate
// (usually 48 kHz) to 16 kHz PCM16 and post ~64 ms chunks to the main
// thread, which forwards them over the WebSocket as binary frames.
class Pcm16Downsampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = (options.processorOptions && options.processorOptions.targetRate) || 16000;
    this.ratio = sampleRate / this.targetRate;
    this.readPos = 0;        // fractional read position into the fifo
    this.fifo = new Float32Array(0);
    this.out = new Int16Array(1024); // ~64 ms at 16 kHz
    this.outPos = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || channel.length === 0) return true;
    // append to fifo
    const joined = new Float32Array(this.fifo.length + channel.length);
    joined.set(this.fifo, 0);
    joined.set(channel, this.fifo.length);
    this.fifo = joined;
    // linear-interpolation resample
    while (this.readPos + 1 < this.fifo.length) {
      const idx = Math.floor(this.readPos);
      const frac = this.readPos - idx;
      const sample = this.fifo[idx] * (1 - frac) + this.fifo[idx + 1] * frac;
      const clamped = Math.max(-1, Math.min(1, sample));
      this.out[this.outPos++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      this.readPos += this.ratio;
      if (this.outPos === this.out.length) {
        this.port.postMessage(this.out.buffer.slice(0));
        this.outPos = 0;
      }
    }
    const keep = Math.floor(this.readPos);
    this.fifo = this.fifo.slice(keep);
    this.readPos -= keep;
    return true;
  }
}

registerProcessor("pcm16-downsampler", Pcm16Downsampler);
