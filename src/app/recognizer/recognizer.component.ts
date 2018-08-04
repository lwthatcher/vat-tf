import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import * as d3 from "d3";
import { Tensor } from '@tensorflow/tfjs';

const DATA_LABELLED = {'pills-blue': true, 'pills-pink': true}


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

  // #region [Constructors]
  constructor() { }

  ngOnInit() {
    this.canvas = document.querySelector("canvas");
    this.context = this.canvas.getContext("2d");

    let dataset = 'pills-blue'

    this.labelled = DATA_LABELLED[dataset]
    this.loadData(dataset);
    console.info('recognizer initialized', this);
  }
  // #endregion

  // #region [Accessors]
  get ready() { return !!this.data }
  // #endregion

  // #region [Event Handlers]

  // #endregion

  // #region [Computation Methods]
  compute() {
    console.log('do something!')
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
    let data = this.parseCSV(raw);
    let labels = []
    if (this.labelled) {
      data = data.map((row) => row.slice(0,-1));
      labels = data.map((row) => row.slice(-1)[0]);
    }
    this.data = this.toTensor(data);
    if (this.labelled) { this.labels = this.toTensor(labels) }
    console.log('data loaded', this.data, this.labels);
  }

  private parseCSV(data) {
    let asNumber = (d: string[]) => { return d.map((di) => +di) };
    return d3.csvParseRows(data, asNumber);
  }

  private toTensor(data) {
    return tf.tensor(data);
  }
  // #endregion
}
