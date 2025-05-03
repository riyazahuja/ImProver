/-- A total complex shape for three complexes shapes `c₁`, `c₂`, `c₁₂` on three types
`I₁`, `I₂` and `I₁₂` consists of the data and properties that will allow the construction
of a total complex functor `HomologicalComplex₂ C c₁ c₂ ⥤ HomologicalComplex C c₁₂` which
sends `K` to a complex which in degree `i₁₂ : I₁₂` consists of the coproduct
of the `(K.X i₁).X i₂` such that `π ⟨i₁, i₂⟩ = i₁₂`. -/
class TotalComplexShape where
  /-- a map on indices -/
  π : I₁ × I₂ → I₁₂
  /-- the sign of the horizontal differential in the total complex -/
  ε₁ : I₁ × I₂ → ℤˣ
  /-- the sign of the vertical differential in the total complex -/
  ε₂ : I₁ × I₂ → ℤˣ
  rel₁ {i₁ i₁' : I₁} (h : c₁.Rel i₁ i₁') (i₂ : I₂) : c₁₂.Rel (π ⟨i₁, i₂⟩) (π ⟨i₁', i₂⟩)
  rel₂ (i₁ : I₁) {i₂ i₂' : I₂} (h : c₂.Rel i₂ i₂') : c₁₂.Rel (π ⟨i₁, i₂⟩) (π ⟨i₁, i₂'⟩)
  ε₂_ε₁ {i₁ i₁' : I₁} {i₂ i₂' : I₂} (h₁ : c₁.Rel i₁ i₁') (h₂ : c₂.Rel i₂ i₂') :
    ε₂ ⟨i₁, i₂⟩ * ε₁ ⟨i₁, i₂'⟩ = - ε₁ ⟨i₁, i₂⟩ * ε₂ ⟨i₁', i₂⟩


/-- The map `I₁ × I₂ → I₁₂` on indices given by `TotalComplexShape c₁ c₂ c₁₂`. -/
abbrev π (i : I₁ × I₂) : I₁₂ := TotalComplexShape.π c₁ c₂ c₁₂ i


/-- The sign of the horizontal differential in the total complex. -/
abbrev ε₁ (i : I₁ × I₂) : ℤˣ := TotalComplexShape.ε₁ c₁ c₂ c₁₂ i


/-- The sign of the vertical differential in the total complex. -/
abbrev ε₂ (i : I₁ × I₂) : ℤˣ := TotalComplexShape.ε₂ c₁ c₂ c₁₂ i


lemma rel_π₁ {i₁ i₁' : I₁} (h : c₁.Rel i₁ i₁') (i₂ : I₂) :
    c₁₂.Rel (π c₁ c₂ c₁₂ ⟨i₁, i₂⟩) (π c₁ c₂ c₁₂ ⟨i₁', i₂⟩) :=
  TotalComplexShape.rel₁ h i₂


lemma next_π₁ {i₁ i₁' : I₁} (h : c₁.Rel i₁ i₁') (i₂ : I₂) :
    c₁₂.next (π c₁ c₂ c₁₂ ⟨i₁, i₂⟩) = π c₁ c₂ c₁₂ ⟨i₁', i₂⟩ :=
  c₁₂.next_eq' (rel_π₁ c₂ c₁₂ h i₂)


lemma prev_π₁ {i₁ i₁' : I₁} (h : c₁.Rel i₁ i₁') (i₂ : I₂) :
    c₁₂.prev (π c₁ c₂ c₁₂ ⟨i₁', i₂⟩) = π c₁ c₂ c₁₂ ⟨i₁, i₂⟩ :=
  c₁₂.prev_eq' (rel_π₁ c₂ c₁₂ h i₂)


lemma rel_π₂ (i₁ : I₁) {i₂ i₂' : I₂} (h : c₂.Rel i₂ i₂') :
    c₁₂.Rel (π c₁ c₂ c₁₂ ⟨i₁, i₂⟩) (π c₁ c₂ c₁₂ ⟨i₁, i₂'⟩) :=
  TotalComplexShape.rel₂ i₁ h


lemma next_π₂ (i₁ : I₁) {i₂ i₂' : I₂} (h : c₂.Rel i₂ i₂') :
    c₁₂.next (π c₁ c₂ c₁₂ ⟨i₁, i₂⟩) = π c₁ c₂ c₁₂ ⟨i₁, i₂'⟩ :=
  c₁₂.next_eq' (rel_π₂ c₁ c₁₂ i₁ h)


lemma prev_π₂ (i₁ : I₁) {i₂ i₂' : I₂} (h : c₂.Rel i₂ i₂') :
    c₁₂.prev (π c₁ c₂ c₁₂ ⟨i₁, i₂'⟩) = π c₁ c₂ c₁₂ ⟨i₁, i₂⟩ :=
  c₁₂.prev_eq' (rel_π₂ c₁ c₁₂ i₁ h)


lemma ε₂_ε₁ {i₁ i₁' : I₁} {i₂ i₂' : I₂} (h₁ : c₁.Rel i₁ i₁') (h₂ : c₂.Rel i₂ i₂') :
    ε₂ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ * ε₁ c₁ c₂ c₁₂ ⟨i₁, i₂'⟩ =
      - ε₁ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ * ε₂ c₁ c₂ c₁₂ ⟨i₁', i₂⟩ :=
  TotalComplexShape.ε₂_ε₁ h₁ h₂


lemma ε₁_ε₂ {i₁ i₁' : I₁} {i₂ i₂' : I₂} (h₁ : c₁.Rel i₁ i₁') (h₂ : c₂.Rel i₂ i₂') :
    ε₁ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ * ε₂ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ =
      - ε₂ c₁ c₂ c₁₂ ⟨i₁', i₂⟩ * ε₁ c₁ c₂ c₁₂ ⟨i₁, i₂'⟩ :=
  Eq.trans (mul_one _).symm (by
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₁₂ : Type u_4
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₁₂ : ComplexShape I₁₂
      inst✝ : TotalComplexShape c₁ c₂ c₁₂
      i₁ i₁' : I₁
      i₂ i₂' : I₂
      h₁ : c₁.Rel i₁ i₁'
      h₂ : c₂.Rel i₂ i₂'
      ⊢ Eq (HMul.hMul (HMul.hMul (c₁.ε₁ c₂ c₁₂ { fst := i₁, snd := i₂ }) (c₁.ε₂ c₂ c …
    -/
    rw [← Int.units_mul_self (ComplexShape.ε₁ c₁ c₂ c₁₂ (i₁, i₂')), mul_assoc]
    conv_lhs =>
      arg 2
      rw [← mul_assoc, ε₂_ε₁ c₁₂ h₁ h₂]
    rw [neg_mul, neg_mul, neg_mul, mul_neg, neg_inj, ← mul_assoc, ← mul_assoc,
      Int.units_mul_self, one_mul])


/-- If `I` is an additive monoid and `c : ComplexShape I`, `c.TensorSigns` contains the data of
map `ε : I → ℤˣ` and properties which allows the construction of a `TotalComplexShape c c c`. -/
class TensorSigns where
  /-- the signs which appear in the vertical differential of the total complex -/
  ε' : Multiplicative I →* ℤˣ
  rel_add (p q r : I) (hpq : c.Rel p q) : c.Rel (p + r) (q + r)
  add_rel (p q r : I) (hpq : c.Rel p q) : c.Rel (r + p) (r + q)
  ε'_succ (p q : I) (hpq : c.Rel p q) : ε' q = - ε' p


/-- The signs which appear in the vertical differential of the total complex. -/
abbrev ε (i : I) : ℤˣ := TensorSigns.ε' c i


lemma rel_add {p q : I} (hpq : c.Rel p q) (r : I) : c.Rel (p + r) (q + r) :=
  TensorSigns.rel_add _ _ _ hpq


lemma add_rel (r : I) {p q : I} (hpq : c.Rel p q) : c.Rel (r + p) (r + q) :=
  TensorSigns.add_rel _ _ _ hpq


@[simp]
lemma ε_zero : c.ε 0 = 1 := by
  /-
    I : Type u_7
    inst✝¹ : AddMonoid I
    c : ComplexShape I
    inst✝ : c.TensorSigns
    ⊢ Eq (c.ε 0) 1
  -/
  apply MonoidHom.map_one
  /-
    🎉 no goals
  -/


lemma ε_succ {p q : I} (hpq : c.Rel p q) : c.ε q = - c.ε p :=
  TensorSigns.ε'_succ p q hpq


lemma ε_add (p q : I) : c.ε (p + q) = c.ε p * c.ε q := by
  /-
    I : Type u_7
    inst✝¹ : AddMonoid I
    c : ComplexShape I
    inst✝ : c.TensorSigns
    p q : I
    ⊢ Eq (c.ε (HAdd.hAdd p q)) (HMul.hMul (c.ε p) (c.ε q))
  -/
  apply MonoidHom.map_mul
  /-
    🎉 no goals
  -/


lemma next_add (p q : I) (hp : c.Rel p (c.next p)) :
    c.next (p + q) = c.next p + q :=
  c.next_eq' (c.rel_add hp q)


lemma next_add' (p q : I) (hq : c.Rel q (c.next q)) :
    c.next (p + q) = p + c.next q :=
  c.next_eq' (c.add_rel p hq)


@[simps]
instance : TotalComplexShape c c c where
  π := fun ⟨p, q⟩ => p + q
  ε₁ := fun _ => 1
  ε₂ := fun ⟨p, _⟩ => c.ε p
  rel₁ h q := c.rel_add h q
  rel₂ p _ _ h := c.add_rel p h
  ε₂_ε₁ h _ := by
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c✝ : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      I : Type u_7
      inst✝¹ : AddMonoid I
      c : ComplexShape I
      inst✝ : c.TensorSigns
      i₁✝ i₁'✝ i₂✝ i₂'✝ : I
      h : c.Rel i₁✝ i₁'✝
      x✝ : c.Rel i₂✝ i₂'✝
      ⊢ Eq (HMul.hMul ((fun x => ComplexShape.instTotalComplexShape.match_1 (fun x = …
    -/
    dsimp
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c✝ : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      I : Type u_7
      inst✝¹ : AddMonoid I
      c : ComplexShape I
      inst✝ : c.TensorSigns
      i₁✝ i₁'✝ i₂✝ i₂'✝ : I
      h : c.Rel i₁✝ i₁'✝
      x✝ : c.Rel i₂✝ i₂'✝
      ⊢ Eq (HMul.hMul (c.ε i₁✝) 1) (HMul.hMul (-1) (c.ε i₁'✝))
    -/
    rw [neg_mul, one_mul, mul_one, c.ε_succ h, neg_neg]
    /-
      🎉 no goals
    -/


instance : TensorSigns (ComplexShape.down ℕ) where
  ε' := MonoidHom.mk' (fun (i : ℕ) => (-1 : ℤˣ) ^ i) (pow_add (-1 : ℤˣ))
                                        /-
                                          I₁ : Type u_1
                                          I₂ : Type u_2
                                          I₃ : Type u_3
                                          I₁₂ : Type u_4
                                          I₂₃ : Type u_5
                                          J : Type u_6
                                          c₁ : ComplexShape I₁
                                          c₂ : ComplexShape I₂
                                          c₃ : ComplexShape I₃
                                          c₁₂ : ComplexShape I₁₂
                                          c₂₃ : ComplexShape I₂₃
                                          c✝ : ComplexShape J
                                          inst✝² : TotalComplexShape c₁ c₂ c₁₂
                                          I : Type u_7
                                          inst✝¹ : AddMonoid I
                                          c : ComplexShape I
                                          inst✝ : c.TensorSigns
                                          p q r : Nat
                                          hpq : Eq (HAdd.hAdd q 1) p
                                          ⊢ (ComplexShape.down Nat).Rel (HAdd.hAdd p r) (HAdd.hAdd q r)
                                        -/
  rel_add p q r (hpq : q + 1 = p) := by dsimp; omega
                                               /-
                                                 🎉 no goals
                                               -/
                                        /-
                                          I₁ : Type u_1
                                          I₂ : Type u_2
                                          I₃ : Type u_3
                                          I₁₂ : Type u_4
                                          I₂₃ : Type u_5
                                          J : Type u_6
                                          c₁ : ComplexShape I₁
                                          c₂ : ComplexShape I₂
                                          c₃ : ComplexShape I₃
                                          c₁₂ : ComplexShape I₁₂
                                          c₂₃ : ComplexShape I₂₃
                                          c✝ : ComplexShape J
                                          inst✝² : TotalComplexShape c₁ c₂ c₁₂
                                          I : Type u_7
                                          inst✝¹ : AddMonoid I
                                          c : ComplexShape I
                                          inst✝ : c.TensorSigns
                                          p q r : Nat
                                          hpq : Eq (HAdd.hAdd q 1) p
                                          ⊢ (ComplexShape.down Nat).Rel (HAdd.hAdd r p) (HAdd.hAdd r q)
                                        -/
  add_rel p q r (hpq : q + 1 = p) := by dsimp; omega
                                               /-
                                                 🎉 no goals
                                               -/
  ε'_succ := by
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c✝ : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      I : Type u_7
      inst✝¹ : AddMonoid I
      c : ComplexShape I
      inst✝ : c.TensorSigns
      ⊢ ∀ (p q : Nat), (ComplexShape.down Nat).Rel p q → Eq ((MonoidHom.mk' (fun i = …
    -/
    rintro _ q rfl
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c✝ : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      I : Type u_7
      inst✝¹ : AddMonoid I
      c : ComplexShape I
      inst✝ : c.TensorSigns
      q : Nat
      ⊢ Eq ((MonoidHom.mk' (fun i => HPow.hPow (-1) i) ⋯) q) (Neg.neg ((MonoidHom.mk …
    -/
    dsimp
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c✝ : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      I : Type u_7
      inst✝¹ : AddMonoid I
      c : ComplexShape I
      inst✝ : c.TensorSigns
      q : Nat
      ⊢ Eq (HPow.hPow (-1) q) (Neg.neg (HPow.hPow (-1) (HAdd.hAdd q 1)))
    -/
    rw [pow_add, pow_one, mul_neg, mul_one, neg_neg]
    /-
      🎉 no goals
    -/


@[simp]
lemma ε_down_ℕ (n : ℕ) : (ComplexShape.down ℕ).ε n = (-1 : ℤˣ) ^ n := rfl


instance : TensorSigns (ComplexShape.up ℤ) where
  ε' := MonoidHom.mk' Int.negOnePow Int.negOnePow_add
                                        /-
                                          I₁ : Type u_1
                                          I₂ : Type u_2
                                          I₃ : Type u_3
                                          I₁₂ : Type u_4
                                          I₂₃ : Type u_5
                                          J : Type u_6
                                          c₁ : ComplexShape I₁
                                          c₂ : ComplexShape I₂
                                          c₃ : ComplexShape I₃
                                          c₁₂ : ComplexShape I₁₂
                                          c₂₃ : ComplexShape I₂₃
                                          c✝ : ComplexShape J
                                          inst✝² : TotalComplexShape c₁ c₂ c₁₂
                                          I : Type u_7
                                          inst✝¹ : AddMonoid I
                                          c : ComplexShape I
                                          inst✝ : c.TensorSigns
                                          p q r : Int
                                          hpq : Eq (HAdd.hAdd p 1) q
                                          ⊢ (ComplexShape.up Int).Rel (HAdd.hAdd p r) (HAdd.hAdd q r)
                                        -/
  rel_add p q r (hpq : p + 1 = q) := by dsimp; omega
                                               /-
                                                 🎉 no goals
                                               -/
                                        /-
                                          I₁ : Type u_1
                                          I₂ : Type u_2
                                          I₃ : Type u_3
                                          I₁₂ : Type u_4
                                          I₂₃ : Type u_5
                                          J : Type u_6
                                          c₁ : ComplexShape I₁
                                          c₂ : ComplexShape I₂
                                          c₃ : ComplexShape I₃
                                          c₁₂ : ComplexShape I₁₂
                                          c₂₃ : ComplexShape I₂₃
                                          c✝ : ComplexShape J
                                          inst✝² : TotalComplexShape c₁ c₂ c₁₂
                                          I : Type u_7
                                          inst✝¹ : AddMonoid I
                                          c : ComplexShape I
                                          inst✝ : c.TensorSigns
                                          p q r : Int
                                          hpq : Eq (HAdd.hAdd p 1) q
                                          ⊢ (ComplexShape.up Int).Rel (HAdd.hAdd r p) (HAdd.hAdd r q)
                                        -/
  add_rel p q r (hpq : p + 1 = q) := by dsimp; omega
                                               /-
                                                 🎉 no goals
                                               -/
  ε'_succ := by
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c✝ : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      I : Type u_7
      inst✝¹ : AddMonoid I
      c : ComplexShape I
      inst✝ : c.TensorSigns
      ⊢ ∀ (p q : Int), (ComplexShape.up Int).Rel p q → Eq ((MonoidHom.mk' Int.negOne …
    -/
    rintro p _ rfl
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c✝ : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      I : Type u_7
      inst✝¹ : AddMonoid I
      c : ComplexShape I
      inst✝ : c.TensorSigns
      p : Int
      ⊢ Eq ((MonoidHom.mk' Int.negOnePow Int.negOnePow_add) (HAdd.hAdd p 1)) (Neg.ne …
    -/
    dsimp
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c✝ : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      I : Type u_7
      inst✝¹ : AddMonoid I
      c : ComplexShape I
      inst✝ : c.TensorSigns
      p : Int
      ⊢ Eq (HAdd.hAdd p 1).negOnePow (Neg.neg p.negOnePow)
    -/
    rw [Int.negOnePow_succ]
    /-
      🎉 no goals
    -/


@[simp]
lemma ε_up_ℤ (n : ℤ) : (ComplexShape.up ℤ).ε n = n.negOnePow := rfl


/-- When we have six complex shapes `c₁`, `c₂`, `c₃`, `c₁₂`, `c₂₃`, `c`, and total functors
`HomologicalComplex₂ C c₁ c₂ ⥤ HomologicalComplex C c₁₂`,
`HomologicalComplex₂ C c₁₂ c₃ ⥤ HomologicalComplex C c`,
`HomologicalComplex₂ C c₂ c₃ ⥤ HomologicalComplex C c₂₃`,
`HomologicalComplex₂ C c₁ c₂₂₃ ⥤ HomologicalComplex C c`, we get two ways to
compute the total complex of a triple complex in `HomologicalComplex₃ C c₁ c₂ c₃`, then
under this assumption `[Associative c₁ c₂ c₃ c₁₂ c₂₃ c]`, these two complexes
canonically identify (without introducing signs). -/
class Associative : Prop where
  assoc (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) :
    π c₁₂ c₃ c ⟨π c₁ c₂ c₁₂ ⟨i₁, i₂⟩, i₃⟩ = π c₁ c₂₃ c ⟨i₁, π c₂ c₃ c₂₃ ⟨i₂, i₃⟩⟩
  ε₁_eq_mul (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) :
    ε₁ c₁ c₂₃ c (i₁, π c₂ c₃ c₂₃ (i₂, i₃)) =
      ε₁ c₁₂ c₃ c (π c₁ c₂ c₁₂ (i₁, i₂), i₃) * ε₁ c₁ c₂ c₁₂ (i₁, i₂)
  ε₂_ε₁ (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) :
    ε₂ c₁ c₂₃ c (i₁, π c₂ c₃ c₂₃ (i₂, i₃)) * ε₁ c₂ c₃ c₂₃ (i₂, i₃) =
      ε₁ c₁₂ c₃ c (π c₁ c₂ c₁₂ (i₁, i₂), i₃) * ε₂ c₁ c₂ c₁₂ (i₁, i₂)
  ε₂_eq_mul (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) :
    ε₂ c₁₂ c₃ c (π c₁ c₂ c₁₂ (i₁, i₂), i₃) =
      (ε₂ c₁ c₂₃ c (i₁, π c₂ c₃ c₂₃ (i₂, i₃)) * ε₂ c₂ c₃ c₂₃ (i₂, i₃))


lemma assoc (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) :
    π c₁₂ c₃ c ⟨π c₁ c₂ c₁₂ ⟨i₁, i₂⟩, i₃⟩ = π c₁ c₂₃ c ⟨i₁, π c₂ c₃ c₂₃ ⟨i₂, i₃⟩⟩ := by
  /-
    I₁ : Type u_1
    I₂ : Type u_2
    I₃ : Type u_3
    I₁₂ : Type u_4
    I₂₃ : Type u_5
    J : Type u_6
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    c₃ : ComplexShape I₃
    c₁₂ : ComplexShape I₁₂
    c₂₃ : ComplexShape I₂₃
    c : ComplexShape J
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c
    inst✝² : TotalComplexShape c₂ c₃ c₂₃
    inst✝¹ : TotalComplexShape c₁ c₂₃ c
    inst✝ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    ⊢ Eq (c₁₂.π c₃ c { fst := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }, snd := i₃ }) ( …
  -/
  apply Associative.assoc
  /-
    🎉 no goals
  -/


lemma associative_ε₁_eq_mul (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) :
    ε₁ c₁ c₂₃ c (i₁, π c₂ c₃ c₂₃ (i₂, i₃)) =
      ε₁ c₁₂ c₃ c (π c₁ c₂ c₁₂ (i₁, i₂), i₃) * ε₁ c₁ c₂ c₁₂ (i₁, i₂) := by
  /-
    I₁ : Type u_1
    I₂ : Type u_2
    I₃ : Type u_3
    I₁₂ : Type u_4
    I₂₃ : Type u_5
    J : Type u_6
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    c₃ : ComplexShape I₃
    c₁₂ : ComplexShape I₁₂
    c₂₃ : ComplexShape I₂₃
    c : ComplexShape J
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c
    inst✝² : TotalComplexShape c₂ c₃ c₂₃
    inst✝¹ : TotalComplexShape c₁ c₂₃ c
    inst✝ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    ⊢ Eq (c₁.ε₁ c₂₃ c { fst := i₁, snd := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ } })  …
  -/
  apply Associative.ε₁_eq_mul
  /-
    🎉 no goals
  -/


lemma associative_ε₂_ε₁ (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) :
    ε₂ c₁ c₂₃ c (i₁, π c₂ c₃ c₂₃ (i₂, i₃)) * ε₁ c₂ c₃ c₂₃ (i₂, i₃) =
      ε₁ c₁₂ c₃ c (π c₁ c₂ c₁₂ (i₁, i₂), i₃) * ε₂ c₁ c₂ c₁₂ (i₁, i₂) := by
  /-
    I₁ : Type u_1
    I₂ : Type u_2
    I₃ : Type u_3
    I₁₂ : Type u_4
    I₂₃ : Type u_5
    J : Type u_6
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    c₃ : ComplexShape I₃
    c₁₂ : ComplexShape I₁₂
    c₂₃ : ComplexShape I₂₃
    c : ComplexShape J
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c
    inst✝² : TotalComplexShape c₂ c₃ c₂₃
    inst✝¹ : TotalComplexShape c₁ c₂₃ c
    inst✝ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    ⊢ Eq (HMul.hMul (c₁.ε₂ c₂₃ c { fst := i₁, snd := c₂.π c₃ c₂₃ { fst := i₂, snd  …
  -/
  apply Associative.ε₂_ε₁
  /-
    🎉 no goals
  -/


lemma associative_ε₂_eq_mul (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) :
    ε₂ c₁₂ c₃ c (π c₁ c₂ c₁₂ (i₁, i₂), i₃) =
      (ε₂ c₁ c₂₃ c (i₁, π c₂ c₃ c₂₃ (i₂, i₃)) * ε₂ c₂ c₃ c₂₃ (i₂, i₃)) := by
  /-
    I₁ : Type u_1
    I₂ : Type u_2
    I₃ : Type u_3
    I₁₂ : Type u_4
    I₂₃ : Type u_5
    J : Type u_6
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    c₃ : ComplexShape I₃
    c₁₂ : ComplexShape I₁₂
    c₂₃ : ComplexShape I₂₃
    c : ComplexShape J
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c
    inst✝² : TotalComplexShape c₂ c₃ c₂₃
    inst✝¹ : TotalComplexShape c₁ c₂₃ c
    inst✝ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    ⊢ Eq (c₁₂.ε₂ c₃ c { fst := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }, snd := i₃ })  …
  -/
  apply Associative.ε₂_eq_mul
  /-
    🎉 no goals
  -/


/-- The map `I₁ × I₂ × I₃ → j` that is obtained using `TotalComplexShape c₁ c₂ c₁₂`
and `TotalComplexShape c₁₂ c₃ c` when `c₁ : ComplexShape I₁`, `c₂ : ComplexShape I₂`,
`c₃ : ComplexShape I₃`, `c₁₂ : ComplexShape I₁₂` and `c : ComplexShape J`. -/
def r : I₁ × I₂ × I₃ → J := fun ⟨i₁, i₂, i₃⟩ ↦ π c₁₂ c₃ c ⟨π c₁ c₂ c₁₂ ⟨i₁, i₂⟩, i₃⟩


/-- The `GradedObject.BifunctorComp₁₂IndexData` which arises from complex shapes. -/
@[reducible]
def ρ₁₂ : GradedObject.BifunctorComp₁₂IndexData (r c₁ c₂ c₃ c₁₂ c) where
  I₁₂ := I₁₂
  p := π c₁ c₂ c₁₂
  q := π c₁₂ c₃ c
  hpq _ := rfl


/-- The `GradedObject.BifunctorComp₂₃IndexData` which arises from complex shapes. -/
@[reducible]
def ρ₂₃ : GradedObject.BifunctorComp₂₃IndexData (r c₁ c₂ c₃ c₁₂ c) where
  I₂₃ := I₂₃
  p := π c₂ c₃ c₂₃
  q := π c₁ c₂₃ c
  hpq := fun ⟨i₁, i₂, i₃⟩ ↦ (assoc c₁ c₂ c₃ c₁₂ c₂₃ c i₁ i₂ i₃).symm


instance {I : Type*} [AddMonoid I] (c : ComplexShape I) [c.TensorSigns] :
    Associative c c c c c c where
  assoc := add_assoc
                        /-
                          I₁ : Type u_1
                          I₂ : Type u_2
                          I₃ : Type u_3
                          I₁₂ : Type u_4
                          I₂₃ : Type u_5
                          J : Type u_6
                          c₁ : ComplexShape I₁
                          c₂ : ComplexShape I₂
                          c₃ : ComplexShape I₃
                          c₁₂ : ComplexShape I₁₂
                          c₂₃ : ComplexShape I₂₃
                          c✝ : ComplexShape J
                          inst✝² : TotalComplexShape c₁ c₂ c₁₂
                          I : Type u_7
                          inst✝¹ : AddMonoid I
                          c : ComplexShape I
                          inst✝ : c.TensorSigns
                          x✝² x✝¹ x✝ : I
                          ⊢ Eq (c.ε₁ c c { fst := x✝², snd := c.π c c { fst := x✝¹, snd := x✝ } }) (HMul …
                        -/
  ε₁_eq_mul _ _ _ := by dsimp; rw [one_mul]
                               /-
                                 🎉 no goals
                               -/
                    /-
                      I₁ : Type u_1
                      I₂ : Type u_2
                      I₃ : Type u_3
                      I₁₂ : Type u_4
                      I₂₃ : Type u_5
                      J : Type u_6
                      c₁ : ComplexShape I₁
                      c₂ : ComplexShape I₂
                      c₃ : ComplexShape I₃
                      c₁₂ : ComplexShape I₁₂
                      c₂₃ : ComplexShape I₂₃
                      c✝ : ComplexShape J
                      inst✝² : TotalComplexShape c₁ c₂ c₁₂
                      I : Type u_7
                      inst✝¹ : AddMonoid I
                      c : ComplexShape I
                      inst✝ : c.TensorSigns
                      x✝² x✝¹ x✝ : I
                      ⊢ Eq (HMul.hMul (c.ε₂ c c { fst := x✝², snd := c.π c c { fst := x✝¹, snd := x✝ …
                    -/
  ε₂_ε₁ _ _ _ := by dsimp; rw [one_mul, mul_one]
                           /-
                             🎉 no goals
                           -/
                        /-
                          I₁ : Type u_1
                          I₂ : Type u_2
                          I₃ : Type u_3
                          I₁₂ : Type u_4
                          I₂₃ : Type u_5
                          J : Type u_6
                          c₁ : ComplexShape I₁
                          c₂ : ComplexShape I₂
                          c₃ : ComplexShape I₃
                          c₁₂ : ComplexShape I₁₂
                          c₂₃ : ComplexShape I₂₃
                          c✝ : ComplexShape J
                          inst✝² : TotalComplexShape c₁ c₂ c₁₂
                          I : Type u_7
                          inst✝¹ : AddMonoid I
                          c : ComplexShape I
                          inst✝ : c.TensorSigns
                          x✝² x✝¹ x✝ : I
                          ⊢ Eq (c.ε₂ c c { fst := c.π c c { fst := x✝², snd := x✝¹ }, snd := x✝ }) (HMul …
                        -/
  ε₂_eq_mul _ _ _ := by dsimp; rw [ε_add]
                               /-
                                 🎉 no goals
                               -/


/-- A total complex shape symmetry contains the data and properties which allow the
identification of the two total complex functors
`HomologicalComplex₂ C c₁ c₂ ⥤ HomologicalComplex C c₁₂`
and `HomologicalComplex₂ C c₂ c₁ ⥤ HomologicalComplex C c₁₂` via the flip. -/
class TotalComplexShapeSymmetry [TotalComplexShape c₁ c₂ c₁₂] [TotalComplexShape c₂ c₁ c₁₂] where
  symm (i₁ : I₁) (i₂ : I₂) : ComplexShape.π c₂ c₁ c₁₂ ⟨i₂, i₁⟩ = ComplexShape.π c₁ c₂ c₁₂ ⟨i₁, i₂⟩
  /-- the signs involved in the symmetry isomorphism of the total complex -/
  σ (i₁ : I₁) (i₂ : I₂) : ℤˣ
  σ_ε₁ {i₁ i₁' : I₁} (h₁ : c₁.Rel i₁ i₁') (i₂ : I₂) :
    σ i₁ i₂ * ComplexShape.ε₁ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ = ComplexShape.ε₂ c₂ c₁ c₁₂ ⟨i₂, i₁⟩ * σ i₁' i₂
  σ_ε₂ (i₁ : I₁) {i₂ i₂' : I₂} (h₂ : c₂.Rel i₂ i₂') :
    σ i₁ i₂ * ComplexShape.ε₂ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ = ComplexShape.ε₁ c₂ c₁ c₁₂ ⟨i₂, i₁⟩ * σ i₁ i₂'


/-- The signs involved in the symmetry isomorphism of the total complex. -/
abbrev σ (i₁ : I₁) (i₂ : I₂) : ℤˣ := TotalComplexShapeSymmetry.σ c₁ c₂ c₁₂ i₁ i₂


lemma π_symm (i₁ : I₁) (i₂ : I₂) :
    π c₂ c₁ c₁₂ ⟨i₂, i₁⟩ = π c₁ c₂ c₁₂ ⟨i₁, i₂⟩ := by
  /-
    I₁ : Type u_1
    I₂ : Type u_2
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    c₁₂ : ComplexShape I₁₂
    inst✝² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹ : TotalComplexShape c₂ c₁ c₁₂
    inst✝ : TotalComplexShapeSymmetry c₁ c₂ c₁₂
    i₁ : I₁
    i₂ : I₂
    ⊢ Eq (c₂.π c₁ c₁₂ { fst := i₂, snd := i₁ }) (c₁.π c₂ c₁₂ { fst := i₁, snd := i …
  -/
  apply TotalComplexShapeSymmetry.symm
  /-
    🎉 no goals
  -/


/-- The symmetry bijection `(π c₂ c₁ c₁₂ ⁻¹' {j}) ≃ (π c₁ c₂ c₁₂ ⁻¹' {j})`. -/
@[simps]
def symmetryEquiv (j : I₁₂) :
    (π c₂ c₁ c₁₂ ⁻¹' {j}) ≃ (π c₁ c₂ c₁₂ ⁻¹' {j}) where
                                              /-
                                                I₁ : Type u_1
                                                I₂ : Type u_2
                                                I₃ : Type u_3
                                                I₁₂ : Type u_4
                                                I₂₃ : Type u_5
                                                J : Type u_6
                                                c₁ : ComplexShape I₁
                                                c₂ : ComplexShape I₂
                                                c₃ : ComplexShape I₃
                                                c₁₂ : ComplexShape I₁₂
                                                c₂₃ : ComplexShape I₂₃
                                                c : ComplexShape J
                                                inst✝² : TotalComplexShape c₁ c₂ c₁₂
                                                inst✝¹ : TotalComplexShape c₂ c₁ c₁₂
                                                inst✝ : TotalComplexShapeSymmetry c₁ c₂ c₁₂
                                                j : I₁₂
                                                x✝ : ↑(Set.preimage (c₂.π c₁ c₁₂) (Singleton.singleton j))
                                                i₂ : I₂
                                                i₁ : I₁
                                                h : Membership.mem (Set.preimage (c₂.π c₁ c₁₂) (Singleton.singleton j)) { fst  …
                                                ⊢ Membership.mem (Set.preimage (c₁.π c₂ c₁₂) (Singleton.singleton j)) { fst := …
                                              -/
  toFun := fun ⟨⟨i₂, i₁⟩, h⟩ => ⟨⟨i₁, i₂⟩, by simpa [π_symm] using h⟩
                                              /-
                                                🎉 no goals
                                              -/
                                               /-
                                                 I₁ : Type u_1
                                                 I₂ : Type u_2
                                                 I₃ : Type u_3
                                                 I₁₂ : Type u_4
                                                 I₂₃ : Type u_5
                                                 J : Type u_6
                                                 c₁ : ComplexShape I₁
                                                 c₂ : ComplexShape I₂
                                                 c₃ : ComplexShape I₃
                                                 c₁₂ : ComplexShape I₁₂
                                                 c₂₃ : ComplexShape I₂₃
                                                 c : ComplexShape J
                                                 inst✝² : TotalComplexShape c₁ c₂ c₁₂
                                                 inst✝¹ : TotalComplexShape c₂ c₁ c₁₂
                                                 inst✝ : TotalComplexShapeSymmetry c₁ c₂ c₁₂
                                                 j : I₁₂
                                                 x✝ : ↑(Set.preimage (c₁.π c₂ c₁₂) (Singleton.singleton j))
                                                 i₁ : I₁
                                                 i₂ : I₂
                                                 h : Membership.mem (Set.preimage (c₁.π c₂ c₁₂) (Singleton.singleton j)) { fst  …
                                                 ⊢ Membership.mem (Set.preimage (c₂.π c₁ c₁₂) (Singleton.singleton j)) { fst := …
                                               -/
  invFun := fun ⟨⟨i₁, i₂⟩, h⟩ => ⟨⟨i₂, i₁⟩, by simpa [π_symm] using h⟩
                                               /-
                                                 🎉 no goals
                                               -/
  left_inv _ := rfl
  right_inv _ := rfl


lemma σ_ε₁ {i₁ i₁' : I₁} (h₁ : c₁.Rel i₁ i₁') (i₂ : I₂) :
    σ c₁ c₂ c₁₂ i₁ i₂ * ε₁ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ = ε₂ c₂ c₁ c₁₂ ⟨i₂, i₁⟩ * σ c₁ c₂ c₁₂ i₁' i₂ :=
  TotalComplexShapeSymmetry.σ_ε₁ h₁ i₂


lemma σ_ε₂ (i₁ : I₁) {i₂ i₂' : I₂} (h₂ : c₂.Rel i₂ i₂') :
    σ c₁ c₂ c₁₂ i₁ i₂ * ε₂ c₁ c₂ c₁₂ ⟨i₁, i₂⟩ = ε₁ c₂ c₁ c₁₂ ⟨i₂, i₁⟩ * σ c₁ c₂ c₁₂ i₁ i₂' :=
  TotalComplexShapeSymmetry.σ_ε₂ i₁ h₂


@[simps]
instance : TotalComplexShapeSymmetry (up ℤ) (up ℤ) (up ℤ) where
  symm p q := add_comm q p
  σ p q := (p * q).negOnePow
  σ_ε₁ := by
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : TotalComplexShape c₂ c₁ c₁₂
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c₁₂
      ⊢ ∀ {i₁ i₁' : Int}, (ComplexShape.up Int).Rel i₁ i₁' → ∀ (i₂ : Int), Eq (HMul. …
    -/
    rintro p _ rfl q
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : TotalComplexShape c₂ c₁ c₁₂
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c₁₂
      p q : Int
      ⊢ Eq (HMul.hMul ((fun p q => (HMul.hMul p q).negOnePow) p q) ((ComplexShape.up …
    -/
    dsimp
    rw [mul_one, ← Int.negOnePow_add, add_comm q, add_mul, one_mul, Int.negOnePow_add,
      Int.negOnePow_add, mul_assoc, Int.units_mul_self, mul_one]
  σ_ε₂ := by
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : TotalComplexShape c₂ c₁ c₁₂
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c₁₂
      ⊢ ∀ (i₁ : Int) {i₂ i₂' : Int}, (ComplexShape.up Int).Rel i₂ i₂' → Eq (HMul.hMu …
    -/
    rintro p q _ rfl
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : TotalComplexShape c₂ c₁ c₁₂
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c₁₂
      p q : Int
      ⊢ Eq (HMul.hMul ((fun p q => (HMul.hMul p q).negOnePow) p q) ((ComplexShape.up …
    -/
    dsimp
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : TotalComplexShape c₂ c₁ c₁₂
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c₁₂
      p q : Int
      ⊢ Eq (HMul.hMul (HMul.hMul p q).negOnePow p.negOnePow) (HMul.hMul 1 (HMul.hMul …
    -/
    rw [one_mul, ← Int.negOnePow_add, mul_add, mul_one]
    /-
      🎉 no goals
    -/


/-- The obvious `TotalComplexShapeSymmetry c₂ c₁ c₁₂` deduced from a
`TotalComplexShapeSymmetry c₁ c₂ c₁₂`. -/
def TotalComplexShapeSymmetry.symmetry [TotalComplexShape c₁ c₂ c₁₂]
    [TotalComplexShape c₂ c₁ c₁₂] [TotalComplexShapeSymmetry c₁ c₂ c₁₂] :
    TotalComplexShapeSymmetry c₂ c₁ c₁₂ where
  symm i₂ i₁ := (ComplexShape.π_symm c₁ c₂ c₁₂ i₁ i₂).symm
  σ i₂ i₁ := ComplexShape.σ c₁ c₂ c₁₂ i₁ i₂
  σ_ε₁ {i₂ i₂'} h₂ i₁ := by
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : TotalComplexShape c₂ c₁ c₁₂
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c₁₂
      i₂ i₂' : I₂
      h₂ : c₂.Rel i₂ i₂'
      i₁ : I₁
      ⊢ Eq (HMul.hMul ((fun i₂ i₁ => c₁.σ c₂ c₁₂ i₁ i₂) i₂ i₁) (c₂.ε₁ c₁ c₁₂ { fst : …
    -/
    dsimp
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : TotalComplexShape c₂ c₁ c₁₂
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c₁₂
      i₂ i₂' : I₂
      h₂ : c₂.Rel i₂ i₂'
      i₁ : I₁
      ⊢ Eq (HMul.hMul (c₁.σ c₂ c₁₂ i₁ i₂) (c₂.ε₁ c₁ c₁₂ { fst := i₂, snd := i₁ })) ( …
    -/
    apply mul_right_cancel (b := ComplexShape.ε₂ c₁ c₂ c₁₂ (i₁, i₂))
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : TotalComplexShape c₂ c₁ c₁₂
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c₁₂
      i₂ i₂' : I₂
      h₂ : c₂.Rel i₂ i₂'
      i₁ : I₁
      ⊢ Eq (HMul.hMul (HMul.hMul (c₁.σ c₂ c₁₂ i₁ i₂) (c₂.ε₁ c₁ c₁₂ { fst := i₂, snd  …
    -/
    rw [mul_assoc]
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : TotalComplexShape c₂ c₁ c₁₂
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c₁₂
      i₂ i₂' : I₂
      h₂ : c₂.Rel i₂ i₂'
      i₁ : I₁
      ⊢ Eq (HMul.hMul (c₁.σ c₂ c₁₂ i₁ i₂) (HMul.hMul (c₂.ε₁ c₁ c₁₂ { fst := i₂, snd  …
    -/
    nth_rw 2 [mul_comm]
    rw [← mul_assoc, ComplexShape.σ_ε₂ c₁ c₁₂ i₁ h₂, mul_comm, ← mul_assoc,
      Int.units_mul_self, one_mul, mul_comm, ← mul_assoc, Int.units_mul_self, one_mul]
  σ_ε₂ i₂ i₁ i₁' h₁ := by
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : TotalComplexShape c₂ c₁ c₁₂
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c₁₂
      i₂ : I₂
      i₁ i₁' : I₁
      h₁ : c₁.Rel i₁ i₁'
      ⊢ Eq (HMul.hMul ((fun i₂ i₁ => c₁.σ c₂ c₁₂ i₁ i₂) i₂ i₁) (c₂.ε₂ c₁ c₁₂ { fst : …
    -/
    dsimp
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : TotalComplexShape c₂ c₁ c₁₂
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c₁₂
      i₂ : I₂
      i₁ i₁' : I₁
      h₁ : c₁.Rel i₁ i₁'
      ⊢ Eq (HMul.hMul (c₁.σ c₂ c₁₂ i₁ i₂) (c₂.ε₂ c₁ c₁₂ { fst := i₂, snd := i₁ })) ( …
    -/
    apply mul_right_cancel (b := ComplexShape.ε₁ c₁ c₂ c₁₂ (i₁, i₂))
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : TotalComplexShape c₂ c₁ c₁₂
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c₁₂
      i₂ : I₂
      i₁ i₁' : I₁
      h₁ : c₁.Rel i₁ i₁'
      ⊢ Eq (HMul.hMul (HMul.hMul (c₁.σ c₂ c₁₂ i₁ i₂) (c₂.ε₂ c₁ c₁₂ { fst := i₂, snd  …
    -/
    rw [mul_assoc]
    /-
      I₁ : Type u_1
      I₂ : Type u_2
      I₃ : Type u_3
      I₁₂ : Type u_4
      I₂₃ : Type u_5
      J : Type u_6
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      c₃ : ComplexShape I₃
      c₁₂ : ComplexShape I₁₂
      c₂₃ : ComplexShape I₂₃
      c : ComplexShape J
      inst✝² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹ : TotalComplexShape c₂ c₁ c₁₂
      inst✝ : TotalComplexShapeSymmetry c₁ c₂ c₁₂
      i₂ : I₂
      i₁ i₁' : I₁
      h₁ : c₁.Rel i₁ i₁'
      ⊢ Eq (HMul.hMul (c₁.σ c₂ c₁₂ i₁ i₂) (HMul.hMul (c₂.ε₂ c₁ c₁₂ { fst := i₂, snd  …
    -/
    nth_rw 2 [mul_comm]
    rw [← mul_assoc, ComplexShape.σ_ε₁ c₂ c₁₂ h₁ i₂, mul_comm, ← mul_assoc,
      Int.units_mul_self, one_mul, mul_comm, ← mul_assoc, Int.units_mul_self, one_mul]


/-- This typeclass expresses that the signs given by `[TotalComplexShapeSymmetry c₁ c₂ c₁₂]`
and by `[TotalComplexShapeSymmetry c₂ c₁ c₁₂]` are compatible. -/
class TotalComplexShapeSymmetrySymmetry [TotalComplexShape c₁ c₂ c₁₂]
    [TotalComplexShape c₂ c₁ c₁₂] [TotalComplexShapeSymmetry c₁ c₂ c₁₂]
    [TotalComplexShapeSymmetry c₂ c₁ c₁₂] : Prop where
  σ_symm i₁ i₂ : ComplexShape.σ c₂ c₁ c₁₂ i₂ i₁ = ComplexShape.σ c₁ c₂ c₁₂ i₁ i₂


lemma σ_symm (i₁ : I₁) (i₂ : I₂) :
    σ c₂ c₁ c₁₂ i₂ i₁ = σ c₁ c₂ c₁₂ i₁ i₂ := by
  /-
    I₁ : Type u_1
    I₂ : Type u_2
    I₁₂ : Type u_4
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    c₁₂ : ComplexShape I₁₂
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₂ c₁ c₁₂
    inst✝² : TotalComplexShapeSymmetry c₁ c₂ c₁₂
    inst✝¹ : TotalComplexShapeSymmetry c₂ c₁ c₁₂
    inst✝ : TotalComplexShapeSymmetrySymmetry c₁ c₂ c₁₂
    i₁ : I₁
    i₂ : I₂
    ⊢ Eq (c₂.σ c₁ c₁₂ i₂ i₁) (c₁.σ c₂ c₁₂ i₁ i₂)
  -/
  apply TotalComplexShapeSymmetrySymmetry.σ_symm
  /-
    🎉 no goals
  -/


