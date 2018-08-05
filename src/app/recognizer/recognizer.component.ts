import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import * as d3 from "d3";
import { Tensor } from '@tensorflow/tfjs';
import { softmax } from '@tensorflow/tfjs-layers/dist/exports_layers';

const DATA_LABELLED = {'pills-blue': true, 'pills-pink': true}
const DIMS = 13


@Component({
  selector: 'app-recognizer',
  templateUrl: './recognizer.component.html',
  styleUrls: ['./recognizer.component.css']
})
export class RecognizerComponent implements OnInit {
  // #region [Variables]
  data: Tensor;
  labels: Tensor;
  labelled: boolean;
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  // #endregion

  // #region [Parameters]
  name = 'pills-blue'
  iterations = 10;
  layers = 1;
  hiddenSize = 64;
  batchSize = 32;
  sampleLength = 100;
  output_classes = 3;
  // #endregion

  // #region [Constructors]
  constructor() { }

  ngOnInit() {
    this.canvas = document.querySelector("canvas");
    this.context = this.canvas.getContext("2d");

    this.labelled = DATA_LABELLED[this.name]
    this.loadData(this.name);
    console.info('recognizer initialized', this);
  }
  // #endregion

  // #region [Accessors]
  get ready() { return !!this.data }

  get length() { return this.data.shape[0] }

  get inputShape() { return [this.sampleLength, DIMS] }
  // #endregion

  // #region [Event Handlers]

  // #endregion

  // #region [Computation Methods]
  async compute() {
    console.log('calling compute');
    let model = this.createModel();
    console.log('created model, computing!')
    for (let i = 0; i < this.iterations; i++) {
      let [xData, yData] = this.getSamples();
      let history = await model.fit(xData, yData, {
        batchSize: this.batchSize,
        epochs: 1,
        validationSplit: 0.1
      })
      console.log('epoch')
      let tloss = history.history['loss'][0]
      let tacc = history.history['acc']
      let vloss = history.history['val_loss'][0]
      let vacc = history.history['val_acc']
      console.log('epoch:', tloss, tacc, vloss, vacc);
    }
    console.log('FINISHED!')
  }

  createModel() {
    let model = tf.sequential();
    model.add(tf.layers.gru({
      units: this.hiddenSize,
      recurrentInitializer: 'glorotNormal',
      inputShape: this.inputShape,
      returnSequences: true
    }))
    model.add(tf.layers.gru({
      units: this.hiddenSize,
      recurrentInitializer: 'glorotNormal',
      returnSequences: true
    }));
    // model.add(tf.layers.timeDistributed(
    //   {layer: tf.layers.dense({units: DIMS})}));
    model.add(tf.layers.dense({units: this.output_classes, activation:"softmax"}));
    model.compile({
      loss: tf.losses.huberLoss,
      optimizer: 'adam',
      metrics: ['accuracy']
    });
    console.log('input shape', JSON.stringify(model.inputs[0].shape));
    console.log('output shape', JSON.stringify(model.outputs[0].shape));
    return model;
  }

  getSamples() {
    const n_per_epoch = Math.floor(this.length / (this.sampleLength * 10));
    let inidices = tf.randomUniform([n_per_epoch], 0, this.length-this.sampleLength, 'int32')
    let [samples, labels] = [[], []]
    // @ts-ignore
    for (let idx of inidices.dataSync()) {
      let sample = tf.slice(this.data, idx, this.sampleLength)
      let logits = tf.slice(this.labels, idx, this.sampleLength)
      samples.push(sample)
      labels.push(tf.oneHot(logits.as1D().toInt(), this.output_classes).toFloat())
    }
    let result = [tf.stack(samples), tf.stack(labels)]
    console.log('SAMPLES', result);
    return result
  }
  // #endregion

  // #region [Examples]
  toy_example() {
    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    // Generate some synthetic data for training.
    const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
    const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

    // Train the model using the data.
    model.fit(xs, ys, {epochs: 100}).then(() => {
      // Use the model to do inference on a data point the model hasn't seen before:
      // @ts-ignore
      let result = model.predict(tf.tensor2d([4, 5, 6, 7], [4, 1])) as Tensor;
      console.log('RESULT:', result.dataSync());
    });
  }
  // #endregion

  // #region [Helper Methods]
  private async loadData(dataset) {
    let raw = await d3.text('../../assets/' + dataset + '.csv');
    let [data, labels] = [this.parseCSV(raw), [] ];
    data = data.map((row) => row.slice(0,-1));
    labels = data.map((row) => row.slice(-1)[0]);
    this.data = this.toTensor(data);
    this.labels = this.toTensor(labels);
    console.log('data loaded', this.data, this.labels);
  }

  private parseCSV(data) {
    let asNumber = (d: string[]) => { return d.map((di) => +di) };
    return d3.csvParseRows(data, asNumber);
  }

  private toTensor(data) { return tf.tensor(data) }
  // #endregion
}
