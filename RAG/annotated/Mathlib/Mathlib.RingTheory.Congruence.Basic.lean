instance : SMul α c.Quotient := inferInstanceAs (SMul α c.toCon.Quotient)


@[simp, norm_cast]
theorem coe_smul (a : α) (x : R) : (↑(a • x) : c.Quotient) = a • (x : c.Quotient) :=
  rfl


instance isScalarTower_right [Add R] [MulOneClass R] [SMul α R] [IsScalarTower α R R]
    (c : RingCon R) : IsScalarTower α c.Quotient c.Quotient where
  smul_assoc _ := Quotient.ind₂' fun _ _ => congr_arg Quotient.mk'' <| smul_mul_assoc _ _ _


instance smulCommClass [Add R] [MulOneClass R] [SMul α R] [IsScalarTower α R R]
    [SMulCommClass α R R] (c : RingCon R) : SMulCommClass α c.Quotient c.Quotient where
  smul_comm _ := Quotient.ind₂' fun _ _ => congr_arg Quotient.mk'' <| (mul_smul_comm _ _ _).symm


instance smulCommClass' [Add R] [MulOneClass R] [SMul α R] [IsScalarTower α R R]
    [SMulCommClass R α R] (c : RingCon R) : SMulCommClass c.Quotient α c.Quotient :=
  haveI := SMulCommClass.symm R α R
  SMulCommClass.symm _ _ _


instance [Monoid α] [NonAssocSemiring R] [DistribMulAction α R] [IsScalarTower α R R]
    (c : RingCon R) : DistribMulAction α c.Quotient :=
  { c.toCon.mulAction with
    smul_zero := fun _ => congr_arg toQuotient <| smul_zero _
    smul_add := fun _ => Quotient.ind₂' fun _ _ => congr_arg toQuotient <| smul_add _ _ _ }


instance [Monoid α] [Semiring R] [MulSemiringAction α R] [IsScalarTower α R R] (c : RingCon R) :
    MulSemiringAction α c.Quotient :=
  { smul_one := fun _ => congr_arg toQuotient <| smul_one _
    smul_mul := fun _ => Quotient.ind₂' fun _ _ => congr_arg toQuotient <|
      MulSemiringAction.smul_mul _ _ _ }


/-- For congruence relations `c, d` on a type `M` with multiplication and addition, `c ≤ d` iff
`∀ x y ∈ M`, `x` is related to `y` by `d` if `x` is related to `y` by `c`. -/
instance : LE (RingCon R) where
  le c d := ∀ ⦃x y⦄, c x y → d x y


/-- Definition of `≤` for congruence relations. -/
theorem le_def {c d : RingCon R} : c ≤ d ↔ ∀ {x y}, c x y → d x y :=
  Iff.rfl


/-- The infimum of a set of congruence relations on a given type with multiplication and
addition. -/
instance : InfSet (RingCon R) where
  sInf S :=
    { r := fun x y => ∀ c : RingCon R, c ∈ S → c x y
      iseqv :=
        ⟨fun x c _hc => c.refl x, fun h c hc => c.symm <| h c hc, fun h1 h2 c hc =>
          c.trans (h1 c hc) <| h2 c hc⟩
      add' := fun h1 h2 c hc => c.add (h1 c hc) <| h2 c hc
      mul' := fun h1 h2 c hc => c.mul (h1 c hc) <| h2 c hc }


/-- The infimum of a set of congruence relations is the same as the infimum of the set's image
    under the map to the underlying equivalence relation. -/
theorem sInf_toSetoid (S : Set (RingCon R)) : (sInf S).toSetoid = sInf ((·.toSetoid) '' S) :=
  Setoid.ext fun x y =>
                               /-
                                 R : Type u_2
                                 inst✝¹ : Add R
                                 inst✝ : Mul R
                                 S : Set (RingCon R)
                                 x y : R
                                 h : (InfSet.sInf S).toSetoid x y
                                 r : Setoid R
                                 x✝ : Membership.mem (Set.image (fun x => x.toSetoid) S) r
                                 c : RingCon R
                                 hS : Membership.mem S c
                                 hr : Eq ((fun x => x.toSetoid) c) r
                                 ⊢ r x y
                               -/
    ⟨fun h r ⟨c, hS, hr⟩ => by rw [← hr]; exact h c hS, fun h c hS => h c.toSetoid ⟨c, hS, rfl⟩⟩
                                          /-
                                            🎉 no goals
                                          -/


/-- The infimum of a set of congruence relations is the same as the infimum of the set's image
    under the map to the underlying binary relation. -/
@[simp, norm_cast]
theorem coe_sInf (S : Set (RingCon R)) : ⇑(sInf S) = sInf ((⇑) '' S) := by
  /-
    R : Type u_2
    inst✝¹ : Add R
    inst✝ : Mul R
    S : Set (RingCon R)
    ⊢ Eq (⇑(InfSet.sInf S)) (InfSet.sInf (Set.image DFunLike.coe S))
  -/
  ext; simp only [sInf_image, iInf_apply, iInf_Prop_eq]; rfl
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp, norm_cast]
theorem coe_iInf {ι : Sort*} (f : ι → RingCon R) : ⇑(iInf f) = ⨅ i, ⇑(f i) := by
  /-
    R : Type u_2
    inst✝¹ : Add R
    inst✝ : Mul R
    ι : Sort u_3
    f : ι → RingCon R
    ⊢ Eq (⇑(iInf f)) (iInf fun i => ⇑(f i))
  -/
  rw [iInf, coe_sInf, ← Set.range_comp, sInf_range, Function.comp_def]
  /-
    🎉 no goals
  -/


instance : PartialOrder (RingCon R) where
  le_refl _c _ _ := id
  le_trans _c1 _c2 _c3 h1 h2 _x _y h := h2 <| h1 h
  le_antisymm _c _d hc hd := ext fun _x _y => ⟨fun h => hc h, fun h => hd h⟩


/-- The complete lattice of congruence relations on a given type with multiplication and
addition. -/
instance : CompleteLattice (RingCon R) where
  __ := completeLatticeOfInf (RingCon R) fun s =>
    ⟨fun r hr x y h => (h : ∀ r ∈ s, (r : RingCon R) x y) r hr,
      fun _r hr _x _y h _r' hr' => hr hr' h⟩
  inf c d :=
    { toSetoid := c.toSetoid ⊓ d.toSetoid
      mul' := fun h1 h2 => ⟨c.mul h1.1 h2.1, d.mul h1.2 h2.2⟩
      add' := fun h1 h2 => ⟨c.add h1.1 h2.1, d.add h1.2 h2.2⟩ }
  inf_le_left _ _ := fun _ _ h => h.1
  inf_le_right _ _ := fun _ _ h => h.2
  le_inf _ _ _ hb hc := fun _ _ h => ⟨hb h, hc h⟩
  top :=
    { (⊤ : Setoid R) with
      mul' := fun _ _ => trivial
      add' := fun _ _ => trivial }
  le_top _ := fun _ _ _h => trivial
  bot :=
    { (⊥ : Setoid R) with
      mul' := congr_arg₂ _
      add' := congr_arg₂ _ }
  bot_le c := fun x _y h => h ▸ c.refl x


@[simp, norm_cast]
theorem coe_top : ⇑(⊤ : RingCon R) = ⊤ := rfl


@[simp, norm_cast]
theorem coe_bot : ⇑(⊥ : RingCon R) = Eq := rfl


/-- The infimum of two congruence relations equals the infimum of the underlying binary
operations. -/
@[simp, norm_cast]
theorem coe_inf {c d : RingCon R} : ⇑(c ⊓ d) = ⇑c ⊓ ⇑d := rfl


/-- Definition of the infimum of two congruence relations. -/
theorem inf_iff_and {c d : RingCon R} {x y} : (c ⊓ d) x y ↔ c x y ∧ d x y :=
  Iff.rfl


instance [Nontrivial R] : Nontrivial (RingCon R) where
  exists_pair_ne :=
    let ⟨x, y, ne⟩ := exists_pair_ne R
                                        /-
                                          α : Type u_1
                                          R : Type u_2
                                          inst✝² : Add R
                                          inst✝¹ : Mul R
                                          inst✝ : Nontrivial R
                                          x y : R
                                          ne : Ne x y
                                          ⊢ Ne ((fun x_1 => x_1 x y) Bot.bot) ((fun x_1 => x_1 x y) Top.top)
                                        -/
    ⟨⊥, ⊤, ne_of_apply_ne (· x y) <| by simp [ne]⟩
                                        /-
                                          🎉 no goals
                                        -/


/-- The inductively defined smallest congruence relation containing a binary relation `r` equals
    the infimum of the set of congruence relations containing `r`. -/
theorem ringConGen_eq (r : R → R → Prop) :
    ringConGen r = sInf {s : RingCon R | ∀ x y, r x y → s x y} :=
  le_antisymm
    (fun _x _y H =>
      RingConGen.Rel.recOn H (fun _ _ h _ hs => hs _ _ h) (RingCon.refl _)
        (fun _ => RingCon.symm _) (fun _ _ => RingCon.trans _)
        (fun _ _ h1 h2 c hc => c.add (h1 c hc) <| h2 c hc)
        (fun _ _ h1 h2 c hc => c.mul (h1 c hc) <| h2 c hc))
    (sInf_le fun _ _ => RingConGen.Rel.of _ _)


/-- The smallest congruence relation containing a binary relation `r` is contained in any
    congruence relation containing `r`. -/
theorem ringConGen_le {r : R → R → Prop} {c : RingCon R}
    (h : ∀ x y, r x y → c x y) : ringConGen r ≤ c := by
  /-
    R : Type u_2
    inst✝¹ : Add R
    inst✝ : Mul R
    r : R → R → Prop
    c : RingCon R
    h : ∀ (x y : R), r x y → c x y
    ⊢ LE.le (ringConGen r) c
  -/
  rw [ringConGen_eq]; exact sInf_le h
                      /-
                        🎉 no goals
                      -/


/-- Given binary relations `r, s` with `r` contained in `s`, the smallest congruence relation
    containing `s` contains the smallest congruence relation containing `r`. -/
theorem ringConGen_mono {r s : R → R → Prop} (h : ∀ x y, r x y → s x y) :
    ringConGen r ≤ ringConGen s :=
  ringConGen_le fun x y hr => RingConGen.Rel.of _ _ <| h x y hr


/-- Congruence relations equal the smallest congruence relation in which they are contained. -/
theorem ringConGen_of_ringCon (c : RingCon R) : ringConGen c = c :=
                  /-
                    R : Type u_2
                    inst✝¹ : Add R
                    inst✝ : Mul R
                    c : RingCon R
                    ⊢ LE.le (ringConGen ⇑c) c
                  -/
  le_antisymm (by rw [ringConGen_eq]; exact sInf_le fun _ _ => id) RingConGen.Rel.of
                                      /-
                                        🎉 no goals
                                      -/


/-- The map sending a binary relation to the smallest congruence relation in which it is
    contained is idempotent. -/
theorem ringConGen_idem (r : R → R → Prop) : ringConGen (ringConGen r) = ringConGen r :=
  ringConGen_of_ringCon _


/-- The supremum of congruence relations `c, d` equals the smallest congruence relation containing
    the binary relation '`x` is related to `y` by `c` or `d`'. -/
theorem sup_eq_ringConGen (c d : RingCon R) : c ⊔ d = ringConGen fun x y => c x y ∨ d x y := by
  /-
    R : Type u_2
    inst✝¹ : Add R
    inst✝ : Mul R
    c d : RingCon R
    ⊢ Eq (Max.max c d) (ringConGen fun x y => Or (c x y) (d x y))
  -/
  rw [ringConGen_eq]
  /-
    R : Type u_2
    inst✝¹ : Add R
    inst✝ : Mul R
    c d : RingCon R
    ⊢ Eq (Max.max c d) (InfSet.sInf (setOf fun s => ∀ (x y : R), Or (c x y) (d x y …
  -/
  apply congr_arg sInf
  /-
    R : Type u_2
    inst✝¹ : Add R
    inst✝ : Mul R
    c d : RingCon R
    ⊢ Eq (setOf fun x => And (LE.le c x) (LE.le d x)) (setOf fun s => ∀ (x y : R), …
  -/
  simp only [le_def, or_imp, ← forall_and]
  /-
    🎉 no goals
  -/


/-- The supremum of two congruence relations equals the smallest congruence relation containing
    the supremum of the underlying binary operations. -/
theorem sup_def {c d : RingCon R} : c ⊔ d = ringConGen (⇑c ⊔ ⇑d) := by
  /-
    R : Type u_2
    inst✝¹ : Add R
    inst✝ : Mul R
    c d : RingCon R
    ⊢ Eq (Max.max c d) (ringConGen (Max.max ⇑c ⇑d))
  -/
  rw [sup_eq_ringConGen]; rfl
                          /-
                            🎉 no goals
                          -/


/-- The supremum of a set of congruence relations `S` equals the smallest congruence relation
    containing the binary relation 'there exists `c ∈ S` such that `x` is related to `y` by
    `c`'. -/
theorem sSup_eq_ringConGen (S : Set (RingCon R)) :
    sSup S = ringConGen fun x y => ∃ c : RingCon R, c ∈ S ∧ c x y := by
  /-
    R : Type u_2
    inst✝¹ : Add R
    inst✝ : Mul R
    S : Set (RingCon R)
    ⊢ Eq (SupSet.sSup S) (ringConGen fun x y => Exists fun c => And (Membership.me …
  -/
  rw [ringConGen_eq]
  /-
    R : Type u_2
    inst✝¹ : Add R
    inst✝ : Mul R
    S : Set (RingCon R)
    ⊢ Eq (SupSet.sSup S) (InfSet.sInf (setOf fun s => ∀ (x y : R), (Exists fun c = …
  -/
  apply congr_arg sInf
  /-
    R : Type u_2
    inst✝¹ : Add R
    inst✝ : Mul R
    S : Set (RingCon R)
    ⊢ Eq (upperBounds S) (setOf fun s => ∀ (x y : R), (Exists fun c => And (Member …
  -/
  ext
  /-
    case h
    R : Type u_2
    inst✝¹ : Add R
    inst✝ : Mul R
    S : Set (RingCon R)
    x✝ : RingCon R
    ⊢ Iff (Membership.mem (upperBounds S) x✝) (Membership.mem (setOf fun s => ∀ (x …
  -/
  exact ⟨fun h _ _ ⟨r, hr⟩ => h hr.1 hr.2, fun h r hS _ _ hr => h _ _ ⟨r, hS, hr⟩⟩
  /-
    🎉 no goals
  -/


/-- The supremum of a set of congruence relations is the same as the smallest congruence relation
    containing the supremum of the set's image under the map to the underlying binary relation. -/
theorem sSup_def {S : Set (RingCon R)} :
    sSup S = ringConGen (sSup (@Set.image (RingCon R) (R → R → Prop) (⇑) S)) := by
  /-
    R : Type u_2
    inst✝¹ : Add R
    inst✝ : Mul R
    S : Set (RingCon R)
    ⊢ Eq (SupSet.sSup S) (ringConGen (SupSet.sSup (Set.image DFunLike.coe S)))
  -/
  rw [sSup_eq_ringConGen, sSup_image]
  /-
    R : Type u_2
    inst✝¹ : Add R
    inst✝ : Mul R
    S : Set (RingCon R)
    ⊢ Eq (ringConGen fun x y => Exists fun c => And (Membership.mem S c) (c x y))  …
  -/
  congr with (x y)
  /-
    case e_r.h.h.a
    R : Type u_2
    inst✝¹ : Add R
    inst✝ : Mul R
    S : Set (RingCon R)
    x y : R
    ⊢ Iff (Exists fun c => And (Membership.mem S c) (c x y)) (iSup (fun a => iSup  …
  -/
  simp only [sSup_image, iSup_apply, iSup_Prop_eq, exists_prop, rel_eq_coe]
  /-
    🎉 no goals
  -/


/-- There is a Galois insertion of congruence relations on a type with multiplication and addition
`R` into binary relations on `R`. -/
protected def gi : @GaloisInsertion (R → R → Prop) (RingCon R) _ _ ringConGen (⇑) where
  choice r _h := ringConGen r
  gc _r c :=
    ⟨fun H _ _ h => H <| RingConGen.Rel.of _ _ h, fun H =>
      ringConGen_of_ringCon c ▸ ringConGen_mono H⟩
  le_l_u x := (ringConGen_of_ringCon x).symm ▸ le_refl x
  choice_eq _ _ := rfl


