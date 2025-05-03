@[to_additive (attr := simp)]
protected theorem SemiconjBy.map [MulHomClass F M N] (h : SemiconjBy a x y) (f : F) :
                                       /-
                                         F : Type u_1
                                         M : Type u_2
                                         N : Type u_3
                                         inst✝³ : Mul M
                                         inst✝² : Mul N
                                         a x y : M
                                         inst✝¹ : FunLike F M N
                                         inst✝ : MulHomClass F M N
                                         h : SemiconjBy a x y
                                         f : F
                                         ⊢ SemiconjBy (f a) (f x) (f y)
                                       -/
    SemiconjBy (f a) (f x) (f y) := by simpa only [SemiconjBy, map_mul] using congr_arg f h
                                       /-
                                         🎉 no goals
                                       -/


@[to_additive (attr := simp)]
protected theorem Commute.map [MulHomClass F M N] (h : Commute x y) (f : F) : Commute (f x) (f y) :=
  SemiconjBy.map h f


@[to_additive (attr := simp)]
protected theorem SemiconjBy.of_map [MulHomClass F M N] (f : F) (hf : Function.Injective f)
    (h : SemiconjBy (f a) (f x) (f y)) : SemiconjBy a x y :=
         /-
           F : Type u_1
           M : Type u_2
           N : Type u_3
           inst✝³ : Mul M
           inst✝² : Mul N
           a x y : M
           inst✝¹ : FunLike F M N
           inst✝ : MulHomClass F M N
           f : F
           hf : Function.Injective ⇑f
           h : SemiconjBy (f a) (f x) (f y)
           ⊢ Eq (f (HMul.hMul a x)) (f (HMul.hMul y a))
         -/
  hf (by simpa only [SemiconjBy, map_mul] using h)
         /-
           🎉 no goals
         -/


@[to_additive (attr := simp)]
theorem Commute.of_map [MulHomClass F M N] {f : F} (hf : Function.Injective f)
    (h : Commute (f x) (f y)) : Commute x y :=
         /-
           F : Type u_1
           M : Type u_2
           N : Type u_3
           inst✝³ : Mul M
           inst✝² : Mul N
           x y : M
           inst✝¹ : FunLike F M N
           inst✝ : MulHomClass F M N
           f : F
           hf : Function.Injective ⇑f
           h : Commute (f x) (f y)
           ⊢ Eq (f (HMul.hMul x y)) (f (HMul.hMul y x))
         -/
  hf (by simpa only [map_mul] using h.eq)
         /-
           🎉 no goals
         -/


