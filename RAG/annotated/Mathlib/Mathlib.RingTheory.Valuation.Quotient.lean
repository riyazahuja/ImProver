/-- If `hJ : J ⊆ supp v` then `onQuotVal hJ` is the induced function on `R / J` as a function.
Note: it's just the function; the valuation is `onQuot hJ`. -/
def onQuotVal {J : Ideal R} (hJ : J ≤ supp v) : R ⧸ J → Γ₀ := fun q =>
  Quotient.liftOn' q v fun a b h =>
    calc
                                    /-
                                      R : Type u_1
                                      Γ₀ : Type u_2
                                      inst✝¹ : CommRing R
                                      inst✝ : LinearOrderedCommMonoidWithZero Γ₀
                                      v : Valuation R Γ₀
                                      J : Ideal R
                                      hJ : LE.le J v.supp
                                      q : HasQuotient.Quotient R J
                                      a b : R
                                      h : (Submodule.quotientRel J) a b
                                      ⊢ Eq (v a) (v (HAdd.hAdd b (Neg.neg (HAdd.hAdd (Neg.neg a) b))))
                                    -/
      v a = v (b + -(-a + b)) := by simp
                                    /-
                                      🎉 no goals
                                    -/
      _ = v b :=
        v.map_add_supp b <| (Ideal.neg_mem_iff _).2 <| hJ <| QuotientAddGroup.leftRel_apply.mp h


/-- The extension of valuation `v` on `R` to valuation on `R / J` if `J ⊆ supp v`. -/
def onQuot {J : Ideal R} (hJ : J ≤ supp v) : Valuation (R ⧸ J) Γ₀ where
  toFun := v.onQuotVal hJ
  map_zero' := v.map_zero
  map_one' := v.map_one
  map_mul' xbar ybar := Quotient.ind₂' v.map_mul xbar ybar
  map_add_le_max' xbar ybar := Quotient.ind₂' v.map_add xbar ybar


@[simp]
theorem onQuot_comap_eq {J : Ideal R} (hJ : J ≤ supp v) :
    (v.onQuot hJ).comap (Ideal.Quotient.mk J) = v :=
  ext fun _ => rfl


theorem self_le_supp_comap (J : Ideal R) (v : Valuation (R ⧸ J) Γ₀) :
    J ≤ (v.comap (Ideal.Quotient.mk J)).supp := by
  /-
    R : Type u_1
    Γ₀ : Type u_2
    inst✝¹ : CommRing R
    inst✝ : LinearOrderedCommMonoidWithZero Γ₀
    J : Ideal R
    v : Valuation (HasQuotient.Quotient R J) Γ₀
    ⊢ LE.le J (Valuation.comap (Ideal.Quotient.mk J) v).supp
  -/
  rw [comap_supp, ← Ideal.map_le_iff_le_comap]
  /-
    R : Type u_1
    Γ₀ : Type u_2
    inst✝¹ : CommRing R
    inst✝ : LinearOrderedCommMonoidWithZero Γ₀
    J : Ideal R
    v : Valuation (HasQuotient.Quotient R J) Γ₀
    ⊢ LE.le (Ideal.map (Ideal.Quotient.mk J) J) v.supp
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem comap_onQuot_eq (J : Ideal R) (v : Valuation (R ⧸ J) Γ₀) :
    (v.comap (Ideal.Quotient.mk J)).onQuot (v.self_le_supp_comap J) = v :=
  ext <| by
    /-
      R : Type u_1
      Γ₀ : Type u_2
      inst✝¹ : CommRing R
      inst✝ : LinearOrderedCommMonoidWithZero Γ₀
      J : Ideal R
      v : Valuation (HasQuotient.Quotient R J) Γ₀
      ⊢ ∀ (r : HasQuotient.Quotient R J), Eq (((Valuation.comap (Ideal.Quotient.mk J …
    -/
    rintro ⟨x⟩
    /-
      case mk
      R : Type u_1
      Γ₀ : Type u_2
      inst✝¹ : CommRing R
      inst✝ : LinearOrderedCommMonoidWithZero Γ₀
      J : Ideal R
      v : Valuation (HasQuotient.Quotient R J) Γ₀
      r✝ : HasQuotient.Quotient R J
      x : R
      ⊢ Eq (((Valuation.comap (Ideal.Quotient.mk J) v).onQuot ⋯) (Quot.mk (⇑(Submodu …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The quotient valuation on `R / J` has support `(supp v) / J` if `J ⊆ supp v`. -/
theorem supp_quot {J : Ideal R} (hJ : J ≤ supp v) :
    supp (v.onQuot hJ) = (supp v).map (Ideal.Quotient.mk J) := by
  /-
    R : Type u_1
    Γ₀ : Type u_2
    inst✝¹ : CommRing R
    inst✝ : LinearOrderedCommMonoidWithZero Γ₀
    v : Valuation R Γ₀
    J : Ideal R
    hJ : LE.le J v.supp
    ⊢ Eq (v.onQuot hJ).supp (Ideal.map (Ideal.Quotient.mk J) v.supp)
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_1
      Γ₀ : Type u_2
      inst✝¹ : CommRing R
      inst✝ : LinearOrderedCommMonoidWithZero Γ₀
      v : Valuation R Γ₀
      J : Ideal R
      hJ : LE.le J v.supp
      ⊢ LE.le (v.onQuot hJ).supp (Ideal.map (Ideal.Quotient.mk J) v.supp)
    -/
  · rintro ⟨x⟩ hx
    /-
      case a.mk
      R : Type u_1
      Γ₀ : Type u_2
      inst✝¹ : CommRing R
      inst✝ : LinearOrderedCommMonoidWithZero Γ₀
      v : Valuation R Γ₀
      J : Ideal R
      hJ : LE.le J v.supp
      x✝ : HasQuotient.Quotient R J
      x : R
      hx : Membership.mem (v.onQuot hJ).supp (Quot.mk (⇑(Submodule.quotientRel J)) x)
      ⊢ Membership.mem (Ideal.map (Ideal.Quotient.mk J) v.supp) (Quot.mk (⇑(Submodul …
    -/
    apply Ideal.subset_span
    /-
      case a.mk.a
      R : Type u_1
      Γ₀ : Type u_2
      inst✝¹ : CommRing R
      inst✝ : LinearOrderedCommMonoidWithZero Γ₀
      v : Valuation R Γ₀
      J : Ideal R
      hJ : LE.le J v.supp
      x✝ : HasQuotient.Quotient R J
      x : R
      hx : Membership.mem (v.onQuot hJ).supp (Quot.mk (⇑(Submodule.quotientRel J)) x)
      ⊢ Membership.mem (Set.image ⇑(Ideal.Quotient.mk J) ↑v.supp) (Quot.mk (⇑(Submod …
    -/
    exact ⟨x, hx, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u_1
      Γ₀ : Type u_2
      inst✝¹ : CommRing R
      inst✝ : LinearOrderedCommMonoidWithZero Γ₀
      v : Valuation R Γ₀
      J : Ideal R
      hJ : LE.le J v.supp
      ⊢ LE.le (Ideal.map (Ideal.Quotient.mk J) v.supp) (v.onQuot hJ).supp
    -/
  · rw [Ideal.map_le_iff_le_comap]
    /-
      case a
      R : Type u_1
      Γ₀ : Type u_2
      inst✝¹ : CommRing R
      inst✝ : LinearOrderedCommMonoidWithZero Γ₀
      v : Valuation R Γ₀
      J : Ideal R
      hJ : LE.le J v.supp
      ⊢ LE.le v.supp (Ideal.comap (Ideal.Quotient.mk J) (v.onQuot hJ).supp)
    -/
    intro x hx
    /-
      case a
      R : Type u_1
      Γ₀ : Type u_2
      inst✝¹ : CommRing R
      inst✝ : LinearOrderedCommMonoidWithZero Γ₀
      v : Valuation R Γ₀
      J : Ideal R
      hJ : LE.le J v.supp
      x : R
      hx : Membership.mem v.supp x
      ⊢ Membership.mem (Ideal.comap (Ideal.Quotient.mk J) (v.onQuot hJ).supp) x
    -/
    exact hx
    /-
      🎉 no goals
    -/


theorem supp_quot_supp : supp (v.onQuot le_rfl) = 0 := by
  /-
    R : Type u_1
    Γ₀ : Type u_2
    inst✝¹ : CommRing R
    inst✝ : LinearOrderedCommMonoidWithZero Γ₀
    v : Valuation R Γ₀
    ⊢ Eq (v.onQuot ⋯).supp 0
  -/
  rw [supp_quot]
  /-
    R : Type u_1
    Γ₀ : Type u_2
    inst✝¹ : CommRing R
    inst✝ : LinearOrderedCommMonoidWithZero Γ₀
    v : Valuation R Γ₀
    ⊢ Eq (Ideal.map (Ideal.Quotient.mk v.supp) v.supp) 0
  -/
  exact Ideal.map_quotient_self _
  /-
    🎉 no goals
  -/


/-- If `hJ : J ⊆ supp v` then `onQuotVal hJ` is the induced function on `R / J` as a function.
Note: it's just the function; the valuation is `onQuot hJ`. -/
def onQuotVal {J : Ideal R} (hJ : J ≤ supp v) : R ⧸ J → Γ₀ :=
  Valuation.onQuotVal v hJ


/-- The extension of valuation `v` on `R` to valuation on `R / J` if `J ⊆ supp v`. -/
def onQuot {J : Ideal R} (hJ : J ≤ supp v) : AddValuation (R ⧸ J) Γ₀ :=
  Valuation.onQuot v hJ


@[simp]
theorem onQuot_comap_eq {J : Ideal R} (hJ : J ≤ supp v) :
    (v.onQuot hJ).comap (Ideal.Quotient.mk J) = v :=
  Valuation.onQuot_comap_eq v hJ


theorem comap_supp {S : Type*} [CommRing S] (f : S →+* R) :
    supp (v.comap f) = Ideal.comap f v.supp :=
  Valuation.comap_supp v f


theorem self_le_supp_comap (J : Ideal R) (v : AddValuation (R ⧸ J) Γ₀) :
    J ≤ (v.comap (Ideal.Quotient.mk J)).supp :=
  Valuation.self_le_supp_comap J v


@[simp]
theorem comap_onQuot_eq (J : Ideal R) (v : AddValuation (R ⧸ J) Γ₀) :
    (v.comap (Ideal.Quotient.mk J)).onQuot (v.self_le_supp_comap J) = v :=
  Valuation.comap_onQuot_eq J v


/-- The quotient valuation on `R / J` has support `(supp v) / J` if `J ⊆ supp v`. -/
theorem supp_quot {J : Ideal R} (hJ : J ≤ supp v) :
    supp (v.onQuot hJ) = (supp v).map (Ideal.Quotient.mk J) :=
  Valuation.supp_quot v hJ


theorem supp_quot_supp : supp ((Valuation.onQuot v) le_rfl) = 0 :=
  Valuation.supp_quot_supp v


