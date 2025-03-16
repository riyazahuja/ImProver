/-- If left-multiplication by any element is cancellative, left-multiplication by `g` is an
embedding. -/
@[to_additive (attr := simps)
      "If left-addition by any element is cancellative, left-addition by `g` is an
        embedding."]
def mulLeftEmbedding [Mul G] [IsLeftCancelMul G] (g : G) : G ↪ G where
  toFun h := g * h
  inj' := mul_right_injective g


/-- If right-multiplication by any element is cancellative, right-multiplication by `g` is an
embedding. -/
@[to_additive (attr := simps)
      "If right-addition by any element is cancellative, right-addition by `g` is an
        embedding."]
def mulRightEmbedding [Mul G] [IsRightCancelMul G] (g : G) : G ↪ G where
  toFun h := h * g
  inj' := mul_left_injective g


@[to_additive]
theorem mulLeftEmbedding_eq_mulRightEmbedding [CommSemigroup G] [IsCancelMul G] (g : G) :
    mulLeftEmbedding g = mulRightEmbedding g := by
  /-
    G : Type u_1
    inst✝¹ : CommSemigroup G
    inst✝ : IsCancelMul G
    g : G
    ⊢ Eq (mulLeftEmbedding g) (mulRightEmbedding g)
  -/
  ext
  /-
    case h
    G : Type u_1
    inst✝¹ : CommSemigroup G
    inst✝ : IsCancelMul G
    g x✝ : G
    ⊢ Eq ((mulLeftEmbedding g) x✝) ((mulRightEmbedding g) x✝)
  -/
  exact mul_comm _ _
  /-
    🎉 no goals
  -/


