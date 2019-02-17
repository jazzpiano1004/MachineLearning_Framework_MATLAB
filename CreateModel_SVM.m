function ML_model = CreateModel_SVM(trainSet, outputSet, SVMParam_ClassName, SVMParam_kernelType, SVMParam_Nu)
    %% Initialize binary classification model as support vector machine (SVM)
    
%     SVMModel = fitcsvm(trainSet, outputSet, 'Standardize', true, 'KernelFunction', SVMParam_kernelType, ...
%                        'KernelScale', 'auto', 'Nu', SVMParam_Nu, 'ClassNames', SVMParam_ClassName);
    SVMModel = fitcsvm(trainSet, outputSet, 'Standardize', true, 'KernelFunction', SVMParam_kernelType, ...
                       'KernelScale', 'auto', 'Nu', SVMParam_Nu, 'ClassNames', SVMParam_ClassName);

    ML_model = SVMModel;   
end