local notation3 "∀* "(...)", "r:(scoped p => Filter.Eventually p (Ultrafilter.toFilter φ)) => r


local notation "β*" => Germ (φ : Filter α) β


instance instGroupWithZero [GroupWithZero β] : GroupWithZero β* where
  __ := instDivInvMonoid
  __ := instMonoidWithZero
  mul_inv_cancel f := inductionOn f fun f hf ↦ coe_eq.2 <| (φ.em fun y ↦ f y = 0).elim
    (fun H ↦ (hf <| coe_eq.2 H).elim) fun H ↦ H.mono fun _ ↦ mul_inv_cancel₀
                             /-
                               α : Type u
                               β : Type v
                               φ : Ultrafilter α
                               inst✝ : GroupWithZero β
                               ⊢ (↑φ).EventuallyEq ((fun x => Function.comp Inv.inv x) fun x => 0) fun x => 0
                             -/
  inv_zero := coe_eq.2 <| by simp only [Function.comp_def, inv_zero, EventuallyEq.rfl]
                             /-
                               🎉 no goals
                             -/


instance instDivisionSemiring [DivisionSemiring β] : DivisionSemiring β* where
  toSemiring := instSemiring
  __ := instGroupWithZero
  nnqsmul := _
  nnqsmul_def := fun _ _ => rfl


instance instDivisionRing [DivisionRing β] : DivisionRing β* where
  __ := instRing
  __ := instDivisionSemiring
  qsmul := _
  qsmul_def := fun _ _ => rfl


instance instSemifield [Semifield β] : Semifield β* where
  __ := instCommSemiring
  __ := instDivisionSemiring


instance instField [Field β] : Field β* where
  __ := instCommRing
  __ := instDivisionRing


theorem coe_lt [Preorder β] {f g : α → β} : (f : β*) < g ↔ ∀* x, f x < g x := by
  /-
    α : Type u
    β : Type v
    φ : Ultrafilter α
    inst✝ : Preorder β
    f g : α → β
    ⊢ Iff (LT.lt ↑f ↑g) (Filter.Eventually (fun x => LT.lt (f x) (g x)) ↑φ)
  -/
  simp only [lt_iff_le_not_le, eventually_and, coe_le, eventually_not, EventuallyLE]
  /-
    🎉 no goals
  -/


theorem coe_pos [Preorder β] [Zero β] {f : α → β} : 0 < (f : β*) ↔ ∀* x, 0 < f x :=
  coe_lt


theorem const_lt [Preorder β] {x y : β} : x < y → (↑x : β*) < ↑y :=
  coe_lt.mpr ∘ liftRel_const


@[simp, norm_cast]
theorem const_lt_iff [Preorder β] {x y : β} : (↑x : β*) < ↑y ↔ x < y :=
  coe_lt.trans liftRel_const_iff


theorem lt_def [Preorder β] : ((· < ·) : β* → β* → Prop) = LiftRel (· < ·) := by
  /-
    α : Type u
    β : Type v
    φ : Ultrafilter α
    inst✝ : Preorder β
    ⊢ Eq (fun x1 x2 => LT.lt x1 x2) (Filter.Germ.LiftRel fun x1 x2 => LT.lt x1 x2)
  -/
  ext ⟨f⟩ ⟨g⟩
  /-
    case h.mk.h.mk.a
    α : Type u
    β : Type v
    φ : Ultrafilter α
    inst✝ : Preorder β
    x✝¹ : (↑φ).Germ β
    f : α → β
    x✝ : (↑φ).Germ β
    g : α → β
    ⊢ Iff (LT.lt (Quot.mk (⇑((↑φ).germSetoid β)) f) (Quot.mk (⇑((↑φ).germSetoid β) …
  -/
  exact coe_lt
  /-
    🎉 no goals
  -/


instance isTotal [LE β] [IsTotal β (· ≤ ·)] : IsTotal β* (· ≤ ·) :=
  ⟨fun f g =>
    inductionOn₂ f g fun _f _g => eventually_or.1 <| Eventually.of_forall fun _x => total_of _ _ _⟩


open Classical in
/-- If `φ` is an ultrafilter then the ultraproduct is a linear order. -/
noncomputable instance instLinearOrder [LinearOrder β] : LinearOrder β* :=
  Lattice.toLinearOrder _


@[to_additive]
noncomputable instance linearOrderedCommGroup [LinearOrderedCommGroup β] :
    LinearOrderedCommGroup β* where
  __ := instOrderedCommGroup
  __ := instLinearOrder


instance instStrictOrderedSemiring [StrictOrderedSemiring β] : StrictOrderedSemiring β* where
  __ := instOrderedSemiring
  __ := instOrderedAddCancelCommMonoid
  mul_lt_mul_of_pos_left x y z := inductionOn₃ x y z fun _f _g _h hfg hh ↦
    coe_lt.2 <| (coe_lt.1 hh).mp <| (coe_lt.1 hfg).mono fun _a ↦ mul_lt_mul_of_pos_left
  mul_lt_mul_of_pos_right x y z := inductionOn₃ x y z fun _f _g _h hfg hh ↦
    coe_lt.2 <| (coe_lt.1 hh).mp <| (coe_lt.1 hfg).mono fun _a ↦ mul_lt_mul_of_pos_right


instance instStrictOrderedCommSemiring [StrictOrderedCommSemiring β] :
    StrictOrderedCommSemiring β* where
  __ := instStrictOrderedSemiring
  __ := instOrderedCommSemiring


instance instStrictOrderedRing [StrictOrderedRing β] : StrictOrderedRing β* where
  __ := instRing
  __ := instStrictOrderedSemiring
  zero_le_one := const_le zero_le_one
  mul_pos x y := inductionOn₂ x y fun _f _g hf hg ↦
    coe_pos.2 <| (coe_pos.1 hg).mp <| (coe_pos.1 hf).mono fun _x ↦ mul_pos


instance instStrictOrderedCommRing [StrictOrderedCommRing β] : StrictOrderedCommRing β* where
  __ := instStrictOrderedRing
  __ := instOrderedCommRing


noncomputable instance instLinearOrderedRing [LinearOrderedRing β] : LinearOrderedRing β* where
  __ := instStrictOrderedRing
  __ := instLinearOrder


noncomputable instance instLinearOrderedField [LinearOrderedField β] : LinearOrderedField β* where
  __ := instLinearOrderedRing
  __ := instField


noncomputable instance instLinearOrderedCommRing [LinearOrderedCommRing β] :
    LinearOrderedCommRing β* where
  __ := instLinearOrderedRing
  __ := instCommMonoid


theorem max_def [LinearOrder β] (x y : β*) : max x y = map₂ max x y :=
  inductionOn₂ x y fun a b => by
    /-
      α : Type u
      β : Type v
      φ : Ultrafilter α
      inst✝ : LinearOrder β
      x y : (↑φ).Germ β
      a b : α → β
      ⊢ Eq (Max.max ↑a ↑b) (Filter.Germ.map₂ Max.max ↑a ↑b)
    -/
    rcases le_total (a : β*) b with h | h
      /-
        case inl
        α : Type u
        β : Type v
        φ : Ultrafilter α
        inst✝ : LinearOrder β
        x y : (↑φ).Germ β
        a b : α → β
        h : LE.le ↑a ↑b
        ⊢ Eq (Max.max ↑a ↑b) (Filter.Germ.map₂ Max.max ↑a ↑b)
      -/
    · rw [max_eq_right h, map₂_coe, coe_eq]
      /-
        case inl
        α : Type u
        β : Type v
        φ : Ultrafilter α
        inst✝ : LinearOrder β
        x y : (↑φ).Germ β
        a b : α → β
        h : LE.le ↑a ↑b
        ⊢ (↑φ).EventuallyEq b fun x => Max.max (a x) (b x)
      -/
      exact h.mono fun i hi => (max_eq_right hi).symm
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u
        β : Type v
        φ : Ultrafilter α
        inst✝ : LinearOrder β
        x y : (↑φ).Germ β
        a b : α → β
        h : LE.le ↑b ↑a
        ⊢ Eq (Max.max ↑a ↑b) (Filter.Germ.map₂ Max.max ↑a ↑b)
      -/
    · rw [max_eq_left h, map₂_coe, coe_eq]
      /-
        case inr
        α : Type u
        β : Type v
        φ : Ultrafilter α
        inst✝ : LinearOrder β
        x y : (↑φ).Germ β
        a b : α → β
        h : LE.le ↑b ↑a
        ⊢ (↑φ).EventuallyEq a fun x => Max.max (a x) (b x)
      -/
      exact h.mono fun i hi => (max_eq_left hi).symm
      /-
        🎉 no goals
      -/


theorem min_def [K : LinearOrder β] (x y : β*) : min x y = map₂ min x y :=
  inductionOn₂ x y fun a b => by
    /-
      α : Type u
      β : Type v
      φ : Ultrafilter α
      K : LinearOrder β
      x y : (↑φ).Germ β
      a b : α → β
      ⊢ Eq (Min.min ↑a ↑b) (Filter.Germ.map₂ Min.min ↑a ↑b)
    -/
    rcases le_total (a : β*) b with h | h
      /-
        case inl
        α : Type u
        β : Type v
        φ : Ultrafilter α
        K : LinearOrder β
        x y : (↑φ).Germ β
        a b : α → β
        h : LE.le ↑a ↑b
        ⊢ Eq (Min.min ↑a ↑b) (Filter.Germ.map₂ Min.min ↑a ↑b)
      -/
    · rw [min_eq_left h, map₂_coe, coe_eq]
      /-
        case inl
        α : Type u
        β : Type v
        φ : Ultrafilter α
        K : LinearOrder β
        x y : (↑φ).Germ β
        a b : α → β
        h : LE.le ↑a ↑b
        ⊢ (↑φ).EventuallyEq a fun x => Min.min (a x) (b x)
      -/
      exact h.mono fun i hi => (min_eq_left hi).symm
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u
        β : Type v
        φ : Ultrafilter α
        K : LinearOrder β
        x y : (↑φ).Germ β
        a b : α → β
        h : LE.le ↑b ↑a
        ⊢ Eq (Min.min ↑a ↑b) (Filter.Germ.map₂ Min.min ↑a ↑b)
      -/
    · rw [min_eq_right h, map₂_coe, coe_eq]
      /-
        case inr
        α : Type u
        β : Type v
        φ : Ultrafilter α
        K : LinearOrder β
        x y : (↑φ).Germ β
        a b : α → β
        h : LE.le ↑b ↑a
        ⊢ (↑φ).EventuallyEq b fun x => Min.min (a x) (b x)
      -/
      exact h.mono fun i hi => (min_eq_right hi).symm
      /-
        🎉 no goals
      -/


theorem abs_def [LinearOrderedAddCommGroup β] (x : β*) : |x| = map abs x :=
  inductionOn x fun _a => rfl


@[simp]
theorem const_max [LinearOrder β] (x y : β) : (↑(max x y : β) : β*) = max ↑x ↑y := by
  /-
    α : Type u
    β : Type v
    φ : Ultrafilter α
    inst✝ : LinearOrder β
    x y : β
    ⊢ Eq (↑(Max.max x y)) (Max.max ↑x ↑y)
  -/
  rw [max_def, map₂_const]
  /-
    🎉 no goals
  -/


@[simp]
theorem const_min [LinearOrder β] (x y : β) : (↑(min x y : β) : β*) = min ↑x ↑y := by
  /-
    α : Type u
    β : Type v
    φ : Ultrafilter α
    inst✝ : LinearOrder β
    x y : β
    ⊢ Eq (↑(Min.min x y)) (Min.min ↑x ↑y)
  -/
  rw [min_def, map₂_const]
  /-
    🎉 no goals
  -/


@[simp]
theorem const_abs [LinearOrderedAddCommGroup β] (x : β) : (↑|x| : β*) = |↑x| := by
  /-
    α : Type u
    β : Type v
    φ : Ultrafilter α
    inst✝ : LinearOrderedAddCommGroup β
    x : β
    ⊢ Eq (↑(abs x)) (abs ↑x)
  -/
  rw [abs_def, map_const]
  /-
    🎉 no goals
  -/


