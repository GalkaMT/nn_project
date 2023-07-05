# :framed_picture: nn_project
[Image Classification multi-page streamlit app](https://galkamt-nn-project-streamlit-app-m9zubm.streamlit.app/)<br>

- Классификация произвольного изображения с помощью модели Inception (обученной на датасете ImageNet. Использована готовая модель из torchvision.models)<br>
- Классификация изображений котов и собак дообученной моделью ResNet18 (Предобученную часть из torchvision.models с заменой последнего слоя.Все параметры кроме классификационного слоя заморожены)<br>
- Классификация клеток крови (4 типа) дообученной моделью ResNet50 (Предобученную часть из torchvision.models с заменой последнего слоя.Все параметры кроме классификационного слоя заморожены)<br>
- Классификация фастфуда (10 категорий) дообученной моделью DenseNet201 (Предобученную часть из torchvision.models с заменой последнего слоя.Все параметры кроме классификационного слоя заморожены) <br>

Примеры работы приложения: <br>
<img src="./example_pic_1.png" width="500"> <img src="./example_pic_2.png" width="500"> <br>
<img src="./example_pic_3.png" width="500"> <img src="./example_pic_4.png" width="500"> <br>

### Linear team:<br>
[IvaElen](https://github.com/IvaElen) - отвечала за классификацию клеток крови<br>
[GalkaMT](https://github.com/GalkaMT) -отвечала за классификацию произвольных изображений, деплой приложения, оформление репозитория<br>
[AlexeyPratsevityi](https://github.com/AlexeyPratsevityi) - отвечал за классификацию стритфуда и кошечек с собачками<br>

**Стэк:** Streamlit, PyTorch, Pillow
