theorem AddHom.le_map_tsub [Preorder β] [Add β] [Sub β] [OrderedSub β] (f : AddHom α β)
    (hf : Monotone f) (a b : α) : f a - f b ≤ f (a - b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁷ : Preorder α
    inst✝⁶ : Add α
    inst✝⁵ : Sub α
    inst✝⁴ : OrderedSub α
    inst✝³ : Preorder β
    inst✝² : Add β
    inst✝¹ : Sub β
    inst✝ : OrderedSub β
    f : AddHom α β
    hf : Monotone ⇑f
    a b : α
    ⊢ LE.le (HSub.hSub (f a) (f b)) (f (HSub.hSub a b))
  -/
  rw [tsub_le_iff_right, ← f.map_add]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁷ : Preorder α
    inst✝⁶ : Add α
    inst✝⁵ : Sub α
    inst✝⁴ : OrderedSub α
    inst✝³ : Preorder β
    inst✝² : Add β
    inst✝¹ : Sub β
    inst✝ : OrderedSub β
    f : AddHom α β
    hf : Monotone ⇑f
    a b : α
    ⊢ LE.le (f a) (f (HAdd.hAdd (HSub.hSub a b) b))
  -/
  exact hf le_tsub_add
  /-
    🎉 no goals
  -/


theorem le_mul_tsub {R : Type*} [Distrib R] [Preorder R] [Sub R] [OrderedSub R]
    [MulLeftMono R] {a b c : R} : a * b - a * c ≤ a * (b - c) :=
  (AddHom.mulLeft a).le_map_tsub (monotone_id.const_mul' a) _ _


theorem le_tsub_mul {R : Type*} [CommSemiring R] [Preorder R] [Sub R] [OrderedSub R]
    [MulLeftMono R] {a b c : R} : a * c - b * c ≤ (a - b) * c := by
  /-
    R : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : Preorder R
    inst✝² : Sub R
    inst✝¹ : OrderedSub R
    inst✝ : MulLeftMono R
    a b c : R
    ⊢ LE.le (HSub.hSub (HMul.hMul a c) (HMul.hMul b c)) (HMul.hMul (HSub.hSub a b) …
  -/
  simpa only [mul_comm _ c] using le_mul_tsub
  /-
    🎉 no goals
  -/


/-- An order isomorphism between types with ordered subtraction preserves subtraction provided that
it preserves addition. -/
theorem OrderIso.map_tsub {M N : Type*} [Preorder M] [Add M] [Sub M] [OrderedSub M]
    [PartialOrder N] [Add N] [Sub N] [OrderedSub N] (e : M ≃o N)
    (h_add : ∀ a b, e (a + b) = e a + e b) (a b : M) : e (a - b) = e a - e b := by
  /-
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Preorder M
    inst✝⁶ : Add M
    inst✝⁵ : Sub M
    inst✝⁴ : OrderedSub M
    inst✝³ : PartialOrder N
    inst✝² : Add N
    inst✝¹ : Sub N
    inst✝ : OrderedSub N
    e : OrderIso M N
    h_add : ∀ (a b : M), Eq (e (HAdd.hAdd a b)) (HAdd.hAdd (e a) (e b))
    a b : M
    ⊢ Eq (e (HSub.hSub a b)) (HSub.hSub (e a) (e b))
  -/
  let e_add : M ≃+ N := { e with map_add' := h_add }
  /-
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Preorder M
    inst✝⁶ : Add M
    inst✝⁵ : Sub M
    inst✝⁴ : OrderedSub M
    inst✝³ : PartialOrder N
    inst✝² : Add N
    inst✝¹ : Sub N
    inst✝ : OrderedSub N
    e : OrderIso M N
    h_add : ∀ (a b : M), Eq (e (HAdd.hAdd a b)) (HAdd.hAdd (e a) (e b))
    a b : M
    e_add : AddEquiv M N := { toEquiv := e.toEquiv, map_add' := h_add }
    ⊢ Eq (e (HSub.hSub a b)) (HSub.hSub (e a) (e b))
  -/
  refine le_antisymm ?_ (e_add.toAddHom.le_map_tsub e.monotone a b)
  /-
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Preorder M
    inst✝⁶ : Add M
    inst✝⁵ : Sub M
    inst✝⁴ : OrderedSub M
    inst✝³ : PartialOrder N
    inst✝² : Add N
    inst✝¹ : Sub N
    inst✝ : OrderedSub N
    e : OrderIso M N
    h_add : ∀ (a b : M), Eq (e (HAdd.hAdd a b)) (HAdd.hAdd (e a) (e b))
    a b : M
    e_add : AddEquiv M N := { toEquiv := e.toEquiv, map_add' := h_add }
    ⊢ LE.le (e (HSub.hSub a b)) (HSub.hSub (e a) (e b))
  -/
  suffices e (e.symm (e a) - e.symm (e b)) ≤ e (e.symm (e a - e b)) by simpa
  /-
    M : Type u_3
    N : Type u_4
    inst✝⁷ : Preorder M
    inst✝⁶ : Add M
    inst✝⁵ : Sub M
    inst✝⁴ : OrderedSub M
    inst✝³ : PartialOrder N
    inst✝² : Add N
    inst✝¹ : Sub N
    inst✝ : OrderedSub N
    e : OrderIso M N
    h_add : ∀ (a b : M), Eq (e (HAdd.hAdd a b)) (HAdd.hAdd (e a) (e b))
    a b : M
    e_add : AddEquiv M N := { toEquiv := e.toEquiv, map_add' := h_add }
    ⊢ LE.le (e (HSub.hSub (e.symm (e a)) (e.symm (e b)))) (e (e.symm (HSub.hSub (e …
  -/
  exact e.monotone (e_add.symm.toAddHom.le_map_tsub e.symm.monotone _ _)
  /-
    🎉 no goals
  -/


theorem AddMonoidHom.le_map_tsub [Preorder β] [AddCommMonoid β] [Sub β] [OrderedSub β] (f : α →+ β)
    (hf : Monotone f) (a b : α) : f a - f b ≤ f (a - b) :=
  f.toAddHom.le_map_tsub hf a b


