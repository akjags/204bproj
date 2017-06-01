function xvalLoss = mvpa_analyze(vw, roiName)

if ieNotDefined('roiName')
  roiName = 'VTC';
end
disp(sprintf('Computing cross-validated MVPA for ROI: %s', roiName));
scanNums = [4 5 6];
scanGroup = 'MotionComp_RefScan1';

mv = mv_init(vw, roiName, scanNums, scanGroup);

tSeries = mv.tSeries; % numTimePoints x numVoxels
labels = mv.trials.label; %1 x numTimePoints --> cell of strings
conds = mv.trials.cond; %1 x numTimePoints --> array of numeric value

attLabel = nan(length(conds), 1);
unattLabel = nan(length(conds), 1);

% Remap attended
% 1:FaceA, 2:BodyA, 3:CarA, 4:HouseA, 5:WordA
for i = 1:20
  attLabel(conds==i) = ceil(i/4);
end

% Remap unattended
% 1:FaceU, 2:BodyU, 3:CarU, 4:HouseU, 5:WordU
for i = 1:20
  switch i
    case {5, 9, 13, 17}
      unattLabel(conds==i) = 1;
    case {1, 10, 14, 18}
      unattLabel(conds==i) = 2;
    case {2, 6, 15, 19}
      unattLabel(conds==i) = 3;
    case{3, 7, 11, 20}
      unattLabel(conds==i) = 4;
    case{4, 8, 12, 16}
      unattLabel(conds==i) = 5;
  end
end

attLabel = attLabel(1:end-3);
unattLabel = unattLabel(1:end-3);

anal2 = 1;
if anal2
  xvalLoss = part2Analysis(mv, tSeries, conds, attLabel, unattLabel);
  return
end


categories = {'Face', 'Body', 'Car', 'House', 'Word'};
xvalLoss = nan(2,5);
for ci = 1:5
  al = attLabel==ci;
  svm = fitcsvm(mv.tSeries, al, 'Crossval', 'on');
  loss = kfoldLoss(svm);
  xvalLoss(1,ci) = loss;
  disp(sprintf('%sA Misclassification: %02.02f%%', categories{ci}, 100*loss))

  ul = unattLabel==ci;
  svm = fitcsvm(mv.tSeries, ul, 'Crossval', 'on');
  uLoss = kfoldLoss(svm);
  xvalLoss(2,ci) = uLoss;
  disp(sprintf('%sU Misclassification: %02.02f%%', categories{ci}, 100*uLoss));
end

figure; bar(xvalLoss');
title(sprintf('%s: Within-Condition Crossvalidated Misclassification Rate', roiName));
set(gca, 'XTickLabel', categories);
ylabel('Error');
legend('Attended', 'Unattended');
drawPublishAxis;


function acc = part2Analysis(mv, tSeries, conds, attLabel, unattLabel)

% start with FB/BF
exc = nchoosek(1:5,2);

categories = {'Face', 'Body', 'Car', 'House', 'Word'};

acc = [];
combos = {};
for ei = 1:size(exc,1)
  excluded = exc(ei,:);

  disp(sprintf('Running for excluded %d and %d', excluded(1), excluded(2)));
  combos{end+1} = sprintf('%s-%s', categories{excluded(1)}, categories{excluded(2)});
  traindata = []; trainlabel = []; index = nan(size(tSeries,1),1); count = 0;
  testdata = []; testlabel = []; index2 = nan(size(tSeries,1),1); count2 = 0;
  for i = 1:size(tSeries,1)
    if (any(attLabel(i) == excluded) && ~any(unattLabel(i)==excluded))
      count = count +1;
      traindata(count,:) = tSeries(i,:);
      trainlabel(count) = attLabel(i);
      index(count) = i;
    elseif any(attLabel(i) == excluded) && any(unattLabel(i)==excluded)
      count2 = count2+1;
      testdata(count2,:) = tSeries(i,:);
      testlabel(count2) = attLabel(i);
      index2(count2) = i;
    end
  end
  svm = fitcsvm(traindata, trainlabel');
  label = predict(svm, testdata);
  accuracy = sum(label' == testlabel)/length(testlabel);
  acc(end+1) = accuracy;
  disp(sprintf('Held-out prediction accuracy: %02.02f%%', accuracy*100));

end

figure; bar(acc); hold on;
plot([0 11], [0.5 0.5], ':k');
set(gca, 'XTickLabel', combos);
title(sprintf('%s: Two-class Classification Accuracy', mv.roi.name));
ylabel('Classification Accuracy');
drawPublishAxis;

