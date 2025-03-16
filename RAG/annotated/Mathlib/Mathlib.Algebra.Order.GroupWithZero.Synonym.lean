instance [h : MulZeroClass α] : MulZeroClass αᵒᵈ := h


instance [h : MulZeroOneClass α] : MulZeroOneClass αᵒᵈ := h


instance [Mul α] [Zero α] [h : NoZeroDivisors α] : NoZeroDivisors αᵒᵈ := h


instance [h : SemigroupWithZero α] : SemigroupWithZero αᵒᵈ := h


instance [h : MonoidWithZero α] : MonoidWithZero αᵒᵈ := h


instance [h : CancelMonoidWithZero α] : CancelMonoidWithZero αᵒᵈ := h


instance [h : CommMonoidWithZero α] : CommMonoidWithZero αᵒᵈ := h


instance [h : CancelCommMonoidWithZero α] : CancelCommMonoidWithZero αᵒᵈ := h


instance [h : GroupWithZero α] : GroupWithZero αᵒᵈ := h


instance [h : CommGroupWithZero α] : CommGroupWithZero αᵒᵈ := h


instance [h : MulZeroClass α] : MulZeroClass (Lex α) := h


instance [h : MulZeroOneClass α] : MulZeroOneClass (Lex α) := h


instance [Mul α] [Zero α] [h : NoZeroDivisors α] : NoZeroDivisors (Lex α) := h


instance [h : SemigroupWithZero α] : SemigroupWithZero (Lex α) := h


instance [h : MonoidWithZero α] : MonoidWithZero (Lex α) := h


instance [h : CancelMonoidWithZero α] : CancelMonoidWithZero (Lex α) := h


instance [h : CommMonoidWithZero α] : CommMonoidWithZero (Lex α) := h


instance [h : CancelCommMonoidWithZero α] : CancelCommMonoidWithZero (Lex α) := h


instance [h : GroupWithZero α] : GroupWithZero (Lex α) := h


instance [h : CommGroupWithZero α] : CommGroupWithZero (Lex α) := h

