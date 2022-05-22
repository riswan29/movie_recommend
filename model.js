const tf = require('@tensorflow/tfjs-node');
const movies = require('./data/movielens100k_details.json');

function loadData() {
  const movie_arr = [];
  for (let i = 0; i < movies.length; i++) {
    movie_arr.push([movies[i]['movie_id']]);
  }
  return movie_arr;
}

// Buat fungsi untuk load data disini
async function loadModel() {
  console.log('Loading model...');
  model = await tf.loadLayersModel('file://./model/model.json');
  console.log('Model loaded!');
}


const movie_arr = tf.tensor(loadData());
const movie_len = movies.length;

// Buat fungsi untuk memberi rekomendasi film
let recomendation = [];
exports.recommend = async function(user_id) {
  let user = tf.fill([movie_len], Number(user_id));
  let movie_in_js_array = movie_arr.arraySync();
  await loadModel();
  console.log('recomending for user : $(user_id)');
  pred_tensor = await model.predict([user, movie_arr]).reshape([movie_len]);
  pred = pred_tensor.arraySync();

  let recomendation = [];

  for (let i = 0; i<6;i++ ){
    max =pred_tensor.argMax().arraySync();
    recomendation.push(movies[max]);
    pred.splice(max,1);
    pred_tensor = tf.tensor(pred);
  }
  console.log(recomendation);
  return recomendation;
}