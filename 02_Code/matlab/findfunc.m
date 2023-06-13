function insts = findfunc(fn,pp)
% This function searches the specified path for all occurences of a 
% specified function, prints the file and line occurence to screen and
% stores it in an output cell. NOTE: not recursive
% fn - file name in pp
% pp - path string

    d = dir([pp filesep '*.m']); % list M-files in folder. Not recursive
    c=1;
    insts = {};
    for k = 1:numel(d)
        file = d(k).name;
        disp(file);
        h = getcallinfo(file);
        for n = 1:numel(h) % distinguish calls from subfunctions etc
            name = h(n).functionPrefix;
            lines = h(n).calls.fcnCalls.lines(strcmp(h(n).calls.fcnCalls.names, fn));
            if lines
                disp(['Function ' fn ' is called by function ' name ' at lines ' num2str(lines)])
                insts{c} = {fn,name,num2str(lines)};
                c=c+1;
            end
        end
    end
end

