/-- An odd function on a linear ordered additive commutative group is strictly monotone on the whole
group provided that it is strictly monotone on `Set.Ici 0`. -/
theorem strictMono_of_odd_strictMonoOn_nonneg {f : G → H} (h₁ : ∀ x, f (-x) = -f x)
    (h₂ : StrictMonoOn f (Ici 0)) : StrictMono f := by
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : LinearOrderedAddCommGroup G
    inst✝ : OrderedAddCommGroup H
    f : G → H
    h₁ : ∀ (x : G), Eq (f (Neg.neg x)) (Neg.neg (f x))
    h₂ : StrictMonoOn f (Set.Ici 0)
    ⊢ StrictMono f
  -/
  refine StrictMonoOn.Iic_union_Ici (fun x hx y hy hxy => neg_lt_neg_iff.1 ?_) h₂
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : LinearOrderedAddCommGroup G
    inst✝ : OrderedAddCommGroup H
    f : G → H
    h₁ : ∀ (x : G), Eq (f (Neg.neg x)) (Neg.neg (f x))
    h₂ : StrictMonoOn f (Set.Ici 0)
    x : G
    hx : Membership.mem (Set.Iic 0) x
    y : G
    hy : Membership.mem (Set.Iic 0) y
    hxy : LT.lt x y
    ⊢ LT.lt (Neg.neg (f y)) (Neg.neg (f x))
  -/
  rw [← h₁, ← h₁]
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : LinearOrderedAddCommGroup G
    inst✝ : OrderedAddCommGroup H
    f : G → H
    h₁ : ∀ (x : G), Eq (f (Neg.neg x)) (Neg.neg (f x))
    h₂ : StrictMonoOn f (Set.Ici 0)
    x : G
    hx : Membership.mem (Set.Iic 0) x
    y : G
    hy : Membership.mem (Set.Iic 0) y
    hxy : LT.lt x y
    ⊢ LT.lt (f (Neg.neg y)) (f (Neg.neg x))
  -/
  exact h₂ (neg_nonneg.2 hy) (neg_nonneg.2 hx) (neg_lt_neg hxy)
  /-
    🎉 no goals
  -/


/-- An odd function on a linear ordered additive commutative group is strictly antitone on the whole
group provided that it is strictly antitone on `Set.Ici 0`. -/
theorem strictAnti_of_odd_strictAntiOn_nonneg {f : G → H} (h₁ : ∀ x, f (-x) = -f x)
    (h₂ : StrictAntiOn f (Ici 0)) : StrictAnti f :=
  @strictMono_of_odd_strictMonoOn_nonneg G Hᵒᵈ _ _ _ h₁ h₂


/-- An odd function on a linear ordered additive commutative group is monotone on the whole group
provided that it is monotone on `Set.Ici 0`. -/
theorem monotone_of_odd_of_monotoneOn_nonneg {f : G → H} (h₁ : ∀ x, f (-x) = -f x)
    (h₂ : MonotoneOn f (Ici 0)) : Monotone f := by
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : LinearOrderedAddCommGroup G
    inst✝ : OrderedAddCommGroup H
    f : G → H
    h₁ : ∀ (x : G), Eq (f (Neg.neg x)) (Neg.neg (f x))
    h₂ : MonotoneOn f (Set.Ici 0)
    ⊢ Monotone f
  -/
  refine MonotoneOn.Iic_union_Ici (fun x hx y hy hxy => neg_le_neg_iff.1 ?_) h₂
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : LinearOrderedAddCommGroup G
    inst✝ : OrderedAddCommGroup H
    f : G → H
    h₁ : ∀ (x : G), Eq (f (Neg.neg x)) (Neg.neg (f x))
    h₂ : MonotoneOn f (Set.Ici 0)
    x : G
    hx : Membership.mem (Set.Iic 0) x
    y : G
    hy : Membership.mem (Set.Iic 0) y
    hxy : LE.le x y
    ⊢ LE.le (Neg.neg (f y)) (Neg.neg (f x))
  -/
  rw [← h₁, ← h₁]
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : LinearOrderedAddCommGroup G
    inst✝ : OrderedAddCommGroup H
    f : G → H
    h₁ : ∀ (x : G), Eq (f (Neg.neg x)) (Neg.neg (f x))
    h₂ : MonotoneOn f (Set.Ici 0)
    x : G
    hx : Membership.mem (Set.Iic 0) x
    y : G
    hy : Membership.mem (Set.Iic 0) y
    hxy : LE.le x y
    ⊢ LE.le (f (Neg.neg y)) (f (Neg.neg x))
  -/
  exact h₂ (neg_nonneg.2 hy) (neg_nonneg.2 hx) (neg_le_neg hxy)
  /-
    🎉 no goals
  -/


/-- An odd function on a linear ordered additive commutative group is antitone on the whole group
provided that it is monotone on `Set.Ici 0`. -/
theorem antitone_of_odd_of_monotoneOn_nonneg {f : G → H} (h₁ : ∀ x, f (-x) = -f x)
    (h₂ : AntitoneOn f (Ici 0)) : Antitone f :=
  @monotone_of_odd_of_monotoneOn_nonneg G Hᵒᵈ _ _ _ h₁ h₂

