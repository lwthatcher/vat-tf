import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { Tensor } from '@tensorflow/tfjs';


@Component({
  selector: 'app-recognizer',
  templateUrl: './recognizer.component.html',
  styleUrls: ['./recognizer.component.css']
})
export class RecognizerComponent implements OnInit {

  // #region [Constructors]
  constructor() { }

  ngOnInit() {
  }
  // #endregion

  // #region [Event Handlers]

  // #endregion

  // #region [Computation Methods]
  compute_example() {
    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    // Generate some synthetic data for training.
    const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
    const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

    // Train the model using the data.
    model.fit(xs, ys, {epochs: 100}).then(() => {
      // Use the model to do inference on a data point the model hasn't seen before:
      // @ts-ignore
      let result = model.predict(tf.tensor2d([4, 5, 6, 7], [4, 1])) as Tensor;
      console.log('RESULT:', result.dataSync());
    });
  }
  // #endregion
}
