/* Date:2022/3/25-2022
 * Author:zhaoyufan
 */

//788. 逆序对的数量 https://www.acwing.com/problem/content/790/
package main

import "fmt"

var ans = 0
var tmp = make([]int, 100001)

func merge_sort(q []int, l, r int) {
	if l >= r {
		return
	}
	mid := (l + r) >> 1
	i, j, k := l, mid+1, 0

	//计算完成后，要保证数组变得有序
	merge_sort(q, l, mid)   //排左边
	merge_sort(q, mid+1, r) //排右边

	//此时 l...mid是递增的 mid+1...r是递增的
	for i <= mid && j <= r {
		if q[i] <= q[j] {
			tmp[k] = q[i]
			k++
			i++
		} else {
			//如果q[i]>q[j]，i之后的数必然大于q[j]
			//一共是 1 + （mid-i）
			ans += mid - i + 1
			tmp[k] = q[j]
			k++
			j++
		}
	}
	for i <= mid {
		tmp[k] = q[i]
		k++
		i++

	}
	for j <= r {
		tmp[k] = q[j]
		k++
		j++
	}

	for i, j := l, 0; i <= r; {
		q[i] = tmp[j]
		i++
		j++
	}
}

func main() {
	var n int
	fmt.Scanf("%d", &n)
	q := make([]int, n)
	for i := range q {
		fmt.Scanf("%d", &q[i])
	}
	merge_sort(q, 0, n-1)
	fmt.Println(ans)
}
