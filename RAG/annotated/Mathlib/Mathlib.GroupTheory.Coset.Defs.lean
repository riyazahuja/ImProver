/-- The equivalence relation corresponding to the partition of a group by left cosets
of a subgroup. -/
@[to_additive "The equivalence relation corresponding to the partition of a group by left cosets
 of a subgroup."]
def leftRel : Setoid α :=
  MulAction.orbitRel s.op α


@[to_additive]
theorem leftRel_apply {x y : α} : leftRel s x y ↔ x⁻¹ * y ∈ s :=
  calc
    (∃ a : s.op, y * MulOpposite.unop a = x) ↔ ∃ a : s, y * a = x :=
      s.equivOp.symm.exists_congr_left
    _ ↔ ∃ a : s, x⁻¹ * y = a⁻¹ := by
      /-
        α : Type u_1
        inst✝ : Group α
        s : Subgroup α
        x y : α
        ⊢ Iff (Exists fun a => Eq (HMul.hMul y ↑a) x) (Exists fun a => Eq (HMul.hMul ( …
      -/
      simp only [inv_mul_eq_iff_eq_mul, Subgroup.coe_inv, eq_mul_inv_iff_mul_eq]
      /-
        🎉 no goals
      -/
                          /-
                            α : Type u_1
                            inst✝ : Group α
                            s : Subgroup α
                            x y : α
                            ⊢ Iff (Exists fun a => Eq (HMul.hMul (Inv.inv x) y) ↑(Inv.inv a)) (Membership. …
                          -/
    _ ↔ x⁻¹ * y ∈ s := by simp [exists_inv_mem_iff_exists_mem]
                          /-
                            🎉 no goals
                          -/


@[to_additive]
theorem leftRel_eq : ⇑(leftRel s) = fun x y => x⁻¹ * y ∈ s :=
  funext₂ <| by
    /-
      α : Type u_1
      inst✝ : Group α
      s : Subgroup α
      ⊢ ∀ (a b : α), Eq ((QuotientGroup.leftRel s) a b) (Membership.mem s (HMul.hMul …
    -/
    simp only [eq_iff_iff]
    /-
      α : Type u_1
      inst✝ : Group α
      s : Subgroup α
      ⊢ ∀ (a b : α), Iff ((QuotientGroup.leftRel s) a b) (Membership.mem s (HMul.hMu …
    -/
    apply leftRel_apply
    /-
      🎉 no goals
    -/


@[to_additive]
instance leftRelDecidable [DecidablePred (· ∈ s)] : DecidableRel (leftRel s).r := fun x y => by
  /-
    α : Type u_1
    inst✝¹ : Group α
    s : Subgroup α
    inst✝ : DecidablePred fun x => Membership.mem s x
    x y : α
    ⊢ Decidable ((QuotientGroup.leftRel s) x y)
  -/
  rw [leftRel_eq]
  /-
    α : Type u_1
    inst✝¹ : Group α
    s : Subgroup α
    inst✝ : DecidablePred fun x => Membership.mem s x
    x y : α
    ⊢ Decidable ((fun x y => Membership.mem s (HMul.hMul (Inv.inv x) y)) x y)
  -/
  exact ‹DecidablePred (· ∈ s)› _
  /-
    🎉 no goals
  -/


/-- `α ⧸ s` is the quotient type representing the left cosets of `s`.
  If `s` is a normal subgroup, `α ⧸ s` is a group -/
@[to_additive "`α ⧸ s` is the quotient type representing the left cosets of `s`.  If `s` is a normal
 subgroup, `α ⧸ s` is a group"]
instance instHasQuotientSubgroup : HasQuotient α (Subgroup α) :=
  ⟨fun s => Quotient (leftRel s)⟩


@[to_additive]
instance [DecidablePred (· ∈ s)] : DecidableEq (α ⧸ s) :=
  @Quotient.decidableEq _ _ (leftRelDecidable _)


/-- The equivalence relation corresponding to the partition of a group by right cosets of a
subgroup. -/
@[to_additive "The equivalence relation corresponding to the partition of a group by right cosets
 of a subgroup."]
def rightRel : Setoid α :=
  MulAction.orbitRel s α


@[to_additive]
theorem rightRel_apply {x y : α} : rightRel s x y ↔ y * x⁻¹ ∈ s :=
  calc
    (∃ a : s, (a : α) * y = x) ↔ ∃ a : s, y * x⁻¹ = a⁻¹ := by
      /-
        α : Type u_1
        inst✝ : Group α
        s : Subgroup α
        x y : α
        ⊢ Iff (Exists fun a => Eq (HMul.hMul (↑a) y) x) (Exists fun a => Eq (HMul.hMul …
      -/
      simp only [mul_inv_eq_iff_eq_mul, Subgroup.coe_inv, eq_inv_mul_iff_mul_eq]
      /-
        🎉 no goals
      -/
                          /-
                            α : Type u_1
                            inst✝ : Group α
                            s : Subgroup α
                            x y : α
                            ⊢ Iff (Exists fun a => Eq (HMul.hMul y (Inv.inv x)) ↑(Inv.inv a)) (Membership. …
                          -/
    _ ↔ y * x⁻¹ ∈ s := by simp [exists_inv_mem_iff_exists_mem]
                          /-
                            🎉 no goals
                          -/


@[to_additive]
theorem rightRel_eq : ⇑(rightRel s) = fun x y => y * x⁻¹ ∈ s :=
  funext₂ <| by
    /-
      α : Type u_1
      inst✝ : Group α
      s : Subgroup α
      ⊢ ∀ (a b : α), Eq ((QuotientGroup.rightRel s) a b) (Membership.mem s (HMul.hMu …
    -/
    simp only [eq_iff_iff]
    /-
      α : Type u_1
      inst✝ : Group α
      s : Subgroup α
      ⊢ ∀ (a b : α), Iff ((QuotientGroup.rightRel s) a b) (Membership.mem s (HMul.hM …
    -/
    apply rightRel_apply
    /-
      🎉 no goals
    -/


@[to_additive]
instance rightRelDecidable [DecidablePred (· ∈ s)] : DecidableRel (rightRel s).r := fun x y => by
  /-
    α : Type u_1
    inst✝¹ : Group α
    s : Subgroup α
    inst✝ : DecidablePred fun x => Membership.mem s x
    x y : α
    ⊢ Decidable ((QuotientGroup.rightRel s) x y)
  -/
  rw [rightRel_eq]
  /-
    α : Type u_1
    inst✝¹ : Group α
    s : Subgroup α
    inst✝ : DecidablePred fun x => Membership.mem s x
    x y : α
    ⊢ Decidable ((fun x y => Membership.mem s (HMul.hMul y (Inv.inv x))) x y)
  -/
  exact ‹DecidablePred (· ∈ s)› _
  /-
    🎉 no goals
  -/


/-- Right cosets are in bijection with left cosets. -/
@[to_additive "Right cosets are in bijection with left cosets."]
def quotientRightRelEquivQuotientLeftRel : Quotient (QuotientGroup.rightRel s) ≃ α ⧸ s where
  toFun :=
    Quotient.map' (fun g => g⁻¹) fun a b => by
      /-
        α : Type u_1
        inst✝ : Group α
        s : Subgroup α
        a b : α
        ⊢ (QuotientGroup.rightRel s) a b → (QuotientGroup.leftRel s) ((fun g => Inv.in …
      -/
      rw [leftRel_apply, rightRel_apply]
      /-
        α : Type u_1
        inst✝ : Group α
        s : Subgroup α
        a b : α
        ⊢ Membership.mem s (HMul.hMul b (Inv.inv a)) → Membership.mem s (HMul.hMul (In …
      -/
      exact fun h => (congr_arg (· ∈ s) (by simp [mul_assoc])).mp (s.inv_mem h)
      /-
        🎉 no goals
      -/
      -- Porting note: replace with `by group`
  invFun :=
    Quotient.map' (fun g => g⁻¹) fun a b => by
      /-
        α : Type u_1
        inst✝ : Group α
        s : Subgroup α
        a b : α
        ⊢ (QuotientGroup.leftRel s) a b → (QuotientGroup.rightRel s) ((fun g => Inv.in …
      -/
      rw [leftRel_apply, rightRel_apply]
      /-
        α : Type u_1
        inst✝ : Group α
        s : Subgroup α
        a b : α
        ⊢ Membership.mem s (HMul.hMul (Inv.inv a) b) → Membership.mem s (HMul.hMul ((f …
      -/
      exact fun h => (congr_arg (· ∈ s) (by simp [mul_assoc])).mp (s.inv_mem h)
      /-
        🎉 no goals
      -/
      -- Porting note: replace with `by group`
  left_inv g :=
    Quotient.inductionOn' g fun g =>
      Quotient.sound'
        (by
          /-
            α : Type u_1
            inst✝ : Group α
            s : Subgroup α
            g✝ : Quotient (QuotientGroup.rightRel s)
            g : α
            ⊢ (QuotientGroup.rightRel s) ((fun g => Inv.inv g) ((fun g => Inv.inv g) g)) g
          -/
          simp only [inv_inv]
          /-
            α : Type u_1
            inst✝ : Group α
            s : Subgroup α
            g✝ : Quotient (QuotientGroup.rightRel s)
            g : α
            ⊢ (QuotientGroup.rightRel s) g g
          -/
          exact Quotient.exact' rfl)
          /-
            🎉 no goals
          -/
  right_inv g :=
    Quotient.inductionOn' g fun g =>
      Quotient.sound'
        (by
          /-
            α : Type u_1
            inst✝ : Group α
            s : Subgroup α
            g✝ : HasQuotient.Quotient α s
            g : α
            ⊢ (QuotientGroup.leftRel s) ((fun g => Inv.inv g) ((fun g => Inv.inv g) g)) g
          -/
          simp only [inv_inv]
          /-
            α : Type u_1
            inst✝ : Group α
            s : Subgroup α
            g✝ : HasQuotient.Quotient α s
            g : α
            ⊢ (QuotientGroup.leftRel s) g g
          -/
          exact Quotient.exact' rfl)
          /-
            🎉 no goals
          -/


/-- The canonical map from a group `α` to the quotient `α ⧸ s`. -/
@[to_additive (attr := coe) "The canonical map from an `AddGroup` `α` to the quotient `α ⧸ s`."]
abbrev mk (a : α) : α ⧸ s :=
  Quotient.mk'' a


@[to_additive]
theorem mk_surjective : Function.Surjective <| @mk _ _ s :=
  Quotient.mk''_surjective


@[to_additive (attr := simp)]
lemma range_mk : range (QuotientGroup.mk (s := s)) = univ := range_eq_univ.mpr mk_surjective


@[to_additive (attr := elab_as_elim)]
theorem induction_on {C : α ⧸ s → Prop} (x : α ⧸ s) (H : ∀ z, C (QuotientGroup.mk z)) : C x :=
  Quotient.inductionOn' x H


@[to_additive]
instance : Coe α (α ⧸ s) :=
  ⟨mk⟩


@[to_additive] alias induction_on' := induction_on

-- `alias` doesn't add the deprecation suggestion to the `to_additive` version
-- see https://github.com/leanprover-community/mathlib4/issues/19424

@[to_additive (attr := simp)]
theorem quotient_liftOn_mk {β} (f : α → β) (h) (x : α) : Quotient.liftOn' (x : α ⧸ s) f h = f x :=
  rfl


@[to_additive]
theorem forall_mk {C : α ⧸ s → Prop} : (∀ x : α ⧸ s, C x) ↔ ∀ x : α, C x :=
  mk_surjective.forall


@[to_additive]
theorem exists_mk {C : α ⧸ s → Prop} : (∃ x : α ⧸ s, C x) ↔ ∃ x : α, C x :=
  mk_surjective.exists


@[to_additive]
instance (s : Subgroup α) : Inhabited (α ⧸ s) :=
  ⟨((1 : α) : α ⧸ s)⟩


@[to_additive]
protected theorem eq {a b : α} : (a : α ⧸ s) = b ↔ a⁻¹ * b ∈ s :=
  calc
    _ ↔ leftRel s a b := Quotient.eq''
                /-
                  α : Type u_1
                  inst✝ : Group α
                  s : Subgroup α
                  a b : α
                  ⊢ Iff ((QuotientGroup.leftRel s) a b) (Membership.mem s (HMul.hMul (Inv.inv a) …
                -/
    _ ↔ _ := by rw [leftRel_apply]
                /-
                  🎉 no goals
                -/


@[to_additive (attr := deprecated "No deprecation message was provided." (since := "2024-08-04"))]
alias eq' := QuotientGroup.eq


@[to_additive]
theorem out_eq' (a : α ⧸ s) : mk a.out = a :=
  Quotient.out_eq' a


@[to_additive QuotientAddGroup.mk_out_eq_mul]
theorem mk_out_eq_mul (g : α) : ∃ h : s, (mk g : α ⧸ s).out = g * h :=
                                                                   /-
                                                                     α : Type u_1
                                                                     inst✝ : Group α
                                                                     s : Subgroup α
                                                                     g : α
                                                                     ⊢ Eq (Quotient.out ↑g) (HMul.hMul g ↑⟨HMul.hMul (Inv.inv g) (Quotient.out ↑g), …
                                                                   -/
  ⟨⟨g⁻¹ * (mk g).out, QuotientGroup.eq.mp (mk g).out_eq'.symm⟩, by rw [mul_inv_cancel_left]⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[to_additive QuotientAddGroup.mk_out'_eq_mul]
alias mk_out'_eq_mul := mk_out_eq_mul

-- `alias` doesn't add the deprecation suggestion to the `to_additive` version
-- see https://github.com/leanprover-community/mathlib4/issues/19424

@[to_additive (attr := simp)]
theorem mk_mul_of_mem (a : α) (hb : b ∈ s) : (mk (a * b) : α ⧸ s) = mk a := by
  /-
    α : Type u_1
    inst✝ : Group α
    s : Subgroup α
    b a : α
    hb : Membership.mem s b
    ⊢ Eq ↑(HMul.hMul a b) ↑a
  -/
  rwa [QuotientGroup.eq, mul_inv_rev, inv_mul_cancel_right, s.inv_mem_iff]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem preimage_image_mk (N : Subgroup α) (s : Set α) :
    mk ⁻¹' ((mk : α → α ⧸ N) '' s) = ⋃ x : N, (· * (x : α)) ⁻¹' s := by
  /-
    α : Type u_1
    inst✝ : Group α
    N : Subgroup α
    s : Set α
    ⊢ Eq (Set.preimage QuotientGroup.mk (Set.image QuotientGroup.mk s)) (Set.iUnio …
  -/
  ext x
  simp only [QuotientGroup.eq, SetLike.exists, exists_prop, Set.mem_preimage, Set.mem_iUnion,
    Set.mem_image, ← eq_inv_mul_iff_mul_eq]
  exact
    ⟨fun ⟨y, hs, hN⟩ => ⟨_, N.inv_mem hN, by simpa using hs⟩, fun ⟨z, hz, hxz⟩ =>
      ⟨x * z, hxz, by simpa using hz⟩⟩


@[to_additive]
theorem preimage_image_mk_eq_iUnion_image (N : Subgroup α) (s : Set α) :
    mk ⁻¹' ((mk : α → α ⧸ N) '' s) = ⋃ x : N, (· * (x : α)) '' s := by
  /-
    α : Type u_1
    inst✝ : Group α
    N : Subgroup α
    s : Set α
    ⊢ Eq (Set.preimage QuotientGroup.mk (Set.image QuotientGroup.mk s)) (Set.iUnio …
  -/
  rw [preimage_image_mk, iUnion_congr_of_surjective (·⁻¹) inv_surjective]
  /-
    α : Type u_1
    inst✝ : Group α
    N : Subgroup α
    s : Set α
    ⊢ ∀ (x : Subtype fun x => Membership.mem N x), Eq (Set.image (fun x_1 => HMul. …
  -/
  exact fun x ↦ image_mul_right'
  /-
    🎉 no goals
  -/


@[to_additive]
theorem preimage_image_mk_eq_mul (N : Subgroup α) (s : Set α) :
    mk ⁻¹' ((mk : α → α ⧸ N) '' s) = s * N := by
  /-
    α : Type u_1
    inst✝ : Group α
    N : Subgroup α
    s : Set α
    ⊢ Eq (Set.preimage QuotientGroup.mk (Set.image QuotientGroup.mk s)) (HMul.hMul …
  -/
  rw [preimage_image_mk_eq_iUnion_image, iUnion_subtype, ← image2_mul, ← iUnion_image_right]
  /-
    α : Type u_1
    inst✝ : Group α
    N : Subgroup α
    s : Set α
    ⊢ Eq (Set.iUnion fun x => Set.iUnion fun hx => Set.image (fun x_1 => HMul.hMul …
  -/
  simp only [SetLike.mem_coe]
  /-
    🎉 no goals
  -/


/-- If two subgroups `M` and `N` of `G` are equal, their quotients are in bijection. -/
@[to_additive "If two subgroups `M` and `N` of `G` are equal, their quotients are in bijection."]
def quotientEquivOfEq (h : s = t) : α ⧸ s ≃ α ⧸ t where
  toFun := Quotient.map' id fun _a _b h' => h ▸ h'
  invFun := Quotient.map' id fun _a _b h' => h.symm ▸ h'
  left_inv q := induction_on q fun _g => rfl
  right_inv q := induction_on q fun _g => rfl


theorem quotientEquivOfEq_mk (h : s = t) (a : α) :
    quotientEquivOfEq h (QuotientGroup.mk a) = QuotientGroup.mk a :=
  rfl


