// 创建一个支持并发数和优先级的异步任务调度器
// 要求：支持addTask方法，该方法返回一个Promise，该promise在异步任务执行完毕后resolve
type PromiseFn = (...args: any[]) => Promise<any>; 

type Task = {
  fn: PromiseFn;
  priority: number;
  resolve: (value: unknown) => void;
  reject: (value: unknown) => void;
};

function Scheduler(max: number) {
  const taskList: Task[] = [];
  let executableLeft = max;
  const run = () => {
    if (executableLeft > 0 && taskList.length > 0) {
      executableLeft -= 1;
      const { fn, resolve, reject } = taskList.shift()!;
      fn().then(resolve).catch(reject).finally(() => {
        executableLeft = Math.min(max, executableLeft += 1);
        run();
      });
    }
  }

  return {
    addTask(task: Task) {
      let resolveFn!: (value: unknown) => void;
      let rejectFn!: (value: unknown) => void;

      const promise = new Promise((resolve, reject) => {
        resolveFn = resolve;
        rejectFn = reject;
      });

      taskList.push({ ...task, resolve: resolveFn, reject: rejectFn });
      taskList.sort((a, b) => a.priority - b.priority);
      run();
      return promise;
    },
  };
}