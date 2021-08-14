y = dataa;
%%plotlarda normalizasyon
%%subplot(2,3,1) % 2 satır 2 sutun 1. değer
%%plot(y)
%%title('Table 1.1')

dy= diff(y);

ly= log(y);

n= y(:,1);
n = (n-min(n))./(max(n)-min(n));
m= y(:,2);
m = (m-min(m))./(max(m)-min(m));
b=y(:,3);
b =(b-min(b))./(max(b)-min(b));
c=y(:,4);
c =(c-min(c))./(max(c)-min(c));
d=y(:,5);
d =(d-min(d))./(max(d)-min(d));

Nor=[n,m,b,c,d];

dfl = diff(ly);
subplot(2,3,2)
plot(dfl)
legend('Normalize diff(log)')
title('Table 1.2')
%%ysa

traindata= y(1:80,:);
traininputs =(traindata(:,1:end-1))';
traintarget=(traindata(:,end))';

testdata= y(end-12:end,:);
testinputs= (testdata(:,1:end-1))';
testtarget= (testdata(:,end))';


%%create network
layers=4;
transferfun={'tansig','purelin'};
trainFcn = 'trainbr';
ag= newfit(traininputs,traintarget,layers,transferfun,trainFcn);

trainednetwork= train(ag,traininputs,traintarget);

exam1 = sim(trainednetwork,testinputs);
examtrained= sim(trainednetwork,traininputs);


errors=(exam1-testtarget);

%%plotlayak sonuçları
subplot(2,3,3)

plot (testtarget,'b');
hold on
plot(exam1,'r')
legend('Real','ANN Results')
title('Table 1.3')

%%histogram
subplot(2,3,4)
histfit(errors)
legend('Errors Historgram')
title('Table 1.4')

subplot(2,3,5)
plot(errors)
legend('ERRORS')
sgtitle('Nomalize ANN Results')
view(trainednetwork)