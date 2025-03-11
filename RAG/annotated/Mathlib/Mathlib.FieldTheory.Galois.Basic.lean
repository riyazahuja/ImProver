/-- A field extension E/F is Galois if it is both separable and normal. Note that in mathlib
a separable extension of fields is by definition algebraic. -/
@[stacks 09I0]
class IsGalois : Prop where
  [to_isSeparable : Algebra.IsSeparable F E]
  [to_normal : Normal F E]


theorem isGalois_iff : IsGalois F E ↔ Algebra.IsSeparable F E ∧ Normal F E :=
  ⟨fun h => ⟨h.1, h.2⟩, fun h =>
    { to_isSeparable := h.1
      to_normal := h.2 }⟩


instance self : IsGalois F F :=
  ⟨⟩


theorem integral [IsGalois F E] (x : E) : IsIntegral F x :=
  to_normal.isIntegral x


theorem separable [IsGalois F E] (x : E) : IsSeparable F x :=
  Algebra.IsSeparable.isSeparable F x


theorem splits [IsGalois F E] (x : E) : (minpoly F x).Splits (algebraMap F E) :=
  Normal.splits' x


/-- Let $E$ be a field. Let $G$ be a finite group acting on $E$.
Then the extension $E / E^G$ is Galois. -/
@[stacks 09I3 "first part"]
instance of_fixed_field (G : Type*) [Group G] [Finite G] [MulSemiringAction G E] :
    IsGalois (FixedPoints.subfield G E) E :=
  ⟨⟩


theorem IntermediateField.AdjoinSimple.card_aut_eq_finrank [FiniteDimensional F E] {α : E}
    (hα : IsIntegral F α) (h_sep : IsSeparable F α)
    (h_splits : (minpoly F α).Splits (algebraMap F F⟮α⟯)) :
    Fintype.card (F⟮α⟯ ≃ₐ[F] F⟮α⟯) = finrank F F⟮α⟯ := by
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    α : E
    hα : IsIntegral F α
    h_sep : IsSeparable F α
    h_splits : Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem (I …
    ⊢ Eq (Fintype.card (AlgEquiv F (Subtype fun x => Membership.mem (IntermediateF …
  -/
  letI : Fintype (F⟮α⟯ →ₐ[F] F⟮α⟯) := IntermediateField.fintypeOfAlgHomAdjoinIntegral F hα
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    α : E
    hα : IsIntegral F α
    h_sep : IsSeparable F α
    h_splits : Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem (I …
    this : Fintype (AlgHom F (Subtype fun x => Membership.mem (IntermediateField.a …
    ⊢ Eq (Fintype.card (AlgEquiv F (Subtype fun x => Membership.mem (IntermediateF …
  -/
  rw [IntermediateField.adjoin.finrank hα]
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    α : E
    hα : IsIntegral F α
    h_sep : IsSeparable F α
    h_splits : Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem (I …
    this : Fintype (AlgHom F (Subtype fun x => Membership.mem (IntermediateField.a …
    ⊢ Eq (Fintype.card (AlgEquiv F (Subtype fun x => Membership.mem (IntermediateF …
  -/
  rw [← IntermediateField.card_algHom_adjoin_integral F hα h_sep h_splits]
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    α : E
    hα : IsIntegral F α
    h_sep : IsSeparable F α
    h_splits : Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem (I …
    this : Fintype (AlgHom F (Subtype fun x => Membership.mem (IntermediateField.a …
    ⊢ Eq (Fintype.card (AlgEquiv F (Subtype fun x => Membership.mem (IntermediateF …
  -/
  exact Fintype.card_congr (algEquivEquivAlgHom F F⟮α⟯)
  /-
    🎉 no goals
  -/


/-- Let $E / F$ be a finite extension of fields. If $E$ is Galois over $F$, then
$|\text{Aut}(E/F)| = [E : F]$. -/
@[stacks 09I1 "'only if' part"]
theorem card_aut_eq_finrank [FiniteDimensional F E] [IsGalois F E] :
    Fintype.card (E ≃ₐ[F] E) = finrank F E := by
  /-
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    inst✝¹ : FiniteDimensional F E
    inst✝ : IsGalois F E
    ⊢ Eq (Fintype.card (AlgEquiv F E E)) (Module.finrank F E)
  -/
  cases' Field.exists_primitive_element F E with α hα
  let iso : F⟮α⟯ ≃ₐ[F] E :=
    { toFun := fun e => e.val
      invFun := fun e => ⟨e, by rw [hα]; exact IntermediateField.mem_top⟩
      left_inv := fun _ => by ext; rfl
      right_inv := fun _ => rfl
      map_mul' := fun _ _ => rfl
      map_add' := fun _ _ => rfl
      commutes' := fun _ => rfl }
  /-
    case intro
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    inst✝¹ : FiniteDimensional F E
    inst✝ : IsGalois F E
    α : E
    hα : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    iso : AlgEquiv F (Subtype fun x => Membership.mem (IntermediateField.adjoin F  …
    ⊢ Eq (Fintype.card (AlgEquiv F E E)) (Module.finrank F E)
  -/
  have H : IsIntegral F α := IsGalois.integral F α
  /-
    case intro
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    inst✝¹ : FiniteDimensional F E
    inst✝ : IsGalois F E
    α : E
    hα : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    iso : AlgEquiv F (Subtype fun x => Membership.mem (IntermediateField.adjoin F  …
    H : IsIntegral F α
    ⊢ Eq (Fintype.card (AlgEquiv F E E)) (Module.finrank F E)
  -/
  have h_sep : IsSeparable F α := IsGalois.separable F α
  /-
    case intro
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    inst✝¹ : FiniteDimensional F E
    inst✝ : IsGalois F E
    α : E
    hα : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    iso : AlgEquiv F (Subtype fun x => Membership.mem (IntermediateField.adjoin F  …
    H : IsIntegral F α
    h_sep : IsSeparable F α
    ⊢ Eq (Fintype.card (AlgEquiv F E E)) (Module.finrank F E)
  -/
  have h_splits : (minpoly F α).Splits (algebraMap F E) := IsGalois.splits F α
  replace h_splits : Polynomial.Splits (algebraMap F F⟮α⟯) (minpoly F α) := by
    simpa using
      Polynomial.splits_comp_of_splits (algebraMap F E) iso.symm.toAlgHom.toRingHom h_splits
  /-
    case intro
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    inst✝¹ : FiniteDimensional F E
    inst✝ : IsGalois F E
    α : E
    hα : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    iso : AlgEquiv F (Subtype fun x => Membership.mem (IntermediateField.adjoin F  …
    H : IsIntegral F α
    h_sep : IsSeparable F α
    h_splits : Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem (I …
    ⊢ Eq (Fintype.card (AlgEquiv F E E)) (Module.finrank F E)
  -/
  rw [← LinearEquiv.finrank_eq iso.toLinearEquiv]
  /-
    case intro
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    inst✝¹ : FiniteDimensional F E
    inst✝ : IsGalois F E
    α : E
    hα : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    iso : AlgEquiv F (Subtype fun x => Membership.mem (IntermediateField.adjoin F  …
    H : IsIntegral F α
    h_sep : IsSeparable F α
    h_splits : Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem (I …
    ⊢ Eq (Fintype.card (AlgEquiv F E E)) (Module.finrank F (Subtype fun x => Membe …
  -/
  rw [← IntermediateField.AdjoinSimple.card_aut_eq_finrank F E H h_sep h_splits]
  /-
    case intro
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    inst✝¹ : FiniteDimensional F E
    inst✝ : IsGalois F E
    α : E
    hα : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    iso : AlgEquiv F (Subtype fun x => Membership.mem (IntermediateField.adjoin F  …
    H : IsIntegral F α
    h_sep : IsSeparable F α
    h_splits : Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem (I …
    ⊢ Eq (Fintype.card (AlgEquiv F E E)) (Fintype.card (AlgEquiv F (Subtype fun x  …
  -/
  apply Fintype.card_congr
  /-
    case intro.f
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    inst✝¹ : FiniteDimensional F E
    inst✝ : IsGalois F E
    α : E
    hα : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    iso : AlgEquiv F (Subtype fun x => Membership.mem (IntermediateField.adjoin F  …
    H : IsIntegral F α
    h_sep : IsSeparable F α
    h_splits : Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem (I …
    ⊢ Equiv (AlgEquiv F E E) (AlgEquiv F (Subtype fun x => Membership.mem (Interme …
  -/
  apply Equiv.mk (fun ϕ => iso.trans (ϕ.trans iso.symm)) fun ϕ => iso.symm.trans (ϕ.trans iso)
    /-
      case intro.f.left_inv
      F : Type u_1
      inst✝⁴ : Field F
      E : Type u_2
      inst✝³ : Field E
      inst✝² : Algebra F E
      inst✝¹ : FiniteDimensional F E
      inst✝ : IsGalois F E
      α : E
      hα : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
      iso : AlgEquiv F (Subtype fun x => Membership.mem (IntermediateField.adjoin F  …
      H : IsIntegral F α
      h_sep : IsSeparable F α
      h_splits : Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem (I …
      ⊢ Function.LeftInverse (fun ϕ => iso.symm.trans (ϕ.trans iso)) fun ϕ => iso.tr …
    -/
  · intro ϕ; ext1; simp only [trans_apply, apply_symm_apply]
                   /-
                     🎉 no goals
                   -/
    /-
      case intro.f.right_inv
      F : Type u_1
      inst✝⁴ : Field F
      E : Type u_2
      inst✝³ : Field E
      inst✝² : Algebra F E
      inst✝¹ : FiniteDimensional F E
      inst✝ : IsGalois F E
      α : E
      hα : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
      iso : AlgEquiv F (Subtype fun x => Membership.mem (IntermediateField.adjoin F  …
      H : IsIntegral F α
      h_sep : IsSeparable F α
      h_splits : Polynomial.Splits (algebraMap F (Subtype fun x => Membership.mem (I …
      ⊢ Function.RightInverse (fun ϕ => iso.symm.trans (ϕ.trans iso)) fun ϕ => iso.t …
    -/
  · intro ϕ; ext1; simp only [trans_apply, symm_apply_apply]
                   /-
                     🎉 no goals
                   -/


/-- Let $E / K / F$ be a tower of field extensions.
If $E$ is Galois over $F$, then $E$ is Galois over $K$. -/
@[stacks 09I2]
theorem IsGalois.tower_top_of_isGalois [IsGalois F E] : IsGalois K E :=
  { to_isSeparable := Algebra.isSeparable_tower_top_of_isSeparable F K E
    to_normal := Normal.tower_top_of_normal F K E }


instance (priority := 100) IsGalois.tower_top_intermediateField (K : IntermediateField F E)
    [IsGalois F E] : IsGalois K E :=
  IsGalois.tower_top_of_isGalois F K E


theorem isGalois_iff_isGalois_bot : IsGalois (⊥ : IntermediateField F E) E ↔ IsGalois F E := by
  /-
    F : Type u_1
    E : Type u_3
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    ⊢ Iff (IsGalois (Subtype fun x => Membership.mem Bot.bot x) E) (IsGalois F E)
  -/
  constructor
    /-
      case mp
      F : Type u_1
      E : Type u_3
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ⊢ IsGalois (Subtype fun x => Membership.mem Bot.bot x) E → IsGalois F E
    -/
  · intro h
    /-
      case mp
      F : Type u_1
      E : Type u_3
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      h : IsGalois (Subtype fun x => Membership.mem Bot.bot x) E
      ⊢ IsGalois F E
    -/
    exact IsGalois.tower_top_of_isGalois (⊥ : IntermediateField F E) F E
    /-
      🎉 no goals
    -/
    /-
      case mpr
      F : Type u_1
      E : Type u_3
      inst✝² : Field F
      inst✝¹ : Field E
      inst✝ : Algebra F E
      ⊢ IsGalois F E → IsGalois (Subtype fun x => Membership.mem Bot.bot x) E
    -/
  · intro h; infer_instance
             /-
               🎉 no goals
             -/


theorem IsGalois.of_algEquiv [IsGalois F E] (f : E ≃ₐ[F] E') : IsGalois F E' :=
  { to_isSeparable := Algebra.IsSeparable.of_algHom F E f.symm
    to_normal := Normal.of_algEquiv f }


theorem AlgEquiv.transfer_galois (f : E ≃ₐ[F] E') : IsGalois F E ↔ IsGalois F E' :=
  ⟨fun _ => IsGalois.of_algEquiv f, fun _ => IsGalois.of_algEquiv f.symm⟩


theorem isGalois_iff_isGalois_top : IsGalois F (⊤ : IntermediateField F E) ↔ IsGalois F E :=
  (IntermediateField.topEquiv : (⊤ : IntermediateField F E) ≃ₐ[F] E).transfer_galois


instance isGalois_bot : IsGalois F (⊥ : IntermediateField F E) :=
  (IntermediateField.botEquiv F E).transfer_galois.mpr (IsGalois.self F)


/-- The intermediate field of fixed points fixed by a monoid action that commutes with the
`F`-action on `E`. -/
def FixedPoints.intermediateField (M : Type*) [Monoid M] [MulSemiringAction M E]
    [SMulCommClass M F E] : IntermediateField F E :=
  { FixedPoints.subfield M E with
    carrier := MulAction.fixedPoints M E
    algebraMap_mem' := fun a g => smul_algebraMap g a }


/-- The intermediate field fixed by a subgroup -/
def fixedField : IntermediateField F E :=
  FixedPoints.intermediateField H


theorem mem_fixedField_iff (x) :
    x ∈ fixedField H ↔ ∀ f ∈ H, f x = x := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    H : Subgroup (AlgEquiv F E E)
    x : E
    ⊢ Iff (Membership.mem (IntermediateField.fixedField H) x) (∀ (f : AlgEquiv F E …
  -/
  show x ∈ MulAction.fixedPoints H E ↔ _
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    H : Subgroup (AlgEquiv F E E)
    x : E
    ⊢ Iff (Membership.mem (MulAction.fixedPoints (Subtype fun x => Membership.mem  …
  -/
  simp only [MulAction.mem_fixedPoints, Subtype.forall, Subgroup.mk_smul, AlgEquiv.smul_def]
  /-
    🎉 no goals
  -/


theorem finrank_fixedField_eq_card [FiniteDimensional F E] [DecidablePred (· ∈ H)] :
    finrank (fixedField H) E = Fintype.card H :=
  FixedPoints.finrank_eq_card H E


/-- The subgroup fixing an intermediate field -/
nonrec def fixingSubgroup : Subgroup (E ≃ₐ[F] E) :=
  fixingSubgroup (E ≃ₐ[F] E) (K : Set E)


theorem le_iff_le : K ≤ fixedField H ↔ H ≤ fixingSubgroup K :=
  ⟨fun h g hg x => h (Subtype.mem x) ⟨g, hg⟩, fun h x hx g => h (Subtype.mem g) ⟨x, hx⟩⟩


/-- The fixing subgroup of `K : IntermediateField F E` is isomorphic to `E ≃ₐ[K] E` -/
def fixingSubgroupEquiv : fixingSubgroup K ≃* E ≃ₐ[K] E where
  toFun ϕ := { AlgEquiv.toRingEquiv (ϕ : E ≃ₐ[F] E) with commutes' := ϕ.mem }
  invFun ϕ := ⟨ϕ.restrictScalars _, ϕ.commutes⟩
                   /-
                     F : Type u_1
                     inst✝² : Field F
                     E : Type u_2
                     inst✝¹ : Field E
                     inst✝ : Algebra F E
                     H : Subgroup (AlgEquiv F E E)
                     K : IntermediateField F E
                     x✝ : Subtype fun x => Membership.mem K.fixingSubgroup x
                     ⊢ Eq
                         ((fun ϕ => ⟨AlgEquiv.restrictScalars F ϕ, ⋯⟩)
                           ((fun ϕ =>
                               let __src := (↑ϕ).toRingEquiv;
                               { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯, commutes'  …
                             x✝))
                         x✝
                   -/
  left_inv _ := by ext; rfl
                        /-
                          🎉 no goals
                        -/
                    /-
                      F : Type u_1
                      inst✝² : Field F
                      E : Type u_2
                      inst✝¹ : Field E
                      inst✝ : Algebra F E
                      H : Subgroup (AlgEquiv F E E)
                      K : IntermediateField F E
                      x✝ : AlgEquiv (Subtype fun x => Membership.mem K x) E E
                      ⊢ Eq
                          ((fun ϕ =>
                              let __src := (↑ϕ).toRingEquiv;
                              { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯, commutes' := …
                            ((fun ϕ => ⟨AlgEquiv.restrictScalars F ϕ, ⋯⟩) x✝))
                          x✝
                    -/
  right_inv _ := by ext; rfl
                         /-
                           🎉 no goals
                         -/
                     /-
                       F : Type u_1
                       inst✝² : Field F
                       E : Type u_2
                       inst✝¹ : Field E
                       inst✝ : Algebra F E
                       H : Subgroup (AlgEquiv F E E)
                       K : IntermediateField F E
                       x✝¹ x✝ : Subtype fun x => Membership.mem K.fixingSubgroup x
                       ⊢ Eq
                           ({
                                 toFun := fun ϕ =>
                                   let __src := (↑ϕ).toRingEquiv;
                                   { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯, commutes …
                                 invFun := fun ϕ => ⟨AlgEquiv.restrictScalars F ϕ, ⋯⟩, left_inv := ⋯, …
                             (HMul.hMul x✝¹ x✝))
                           (HMul.hMul
                             ({
                                   toFun := fun ϕ =>
                                     let __src := (↑ϕ).toRingEquiv;
                                     { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯, commut …
                                   invFun := fun ϕ => ⟨AlgEquiv.restrictScalars F ϕ, ⋯⟩, left_inv :=  …
                               x✝¹)
                             ({
                                   toFun := fun ϕ =>
                                     let __src := (↑ϕ).toRingEquiv;
                                     { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯, commut …
                                   invFun := fun ϕ => ⟨AlgEquiv.restrictScalars F ϕ, ⋯⟩, left_inv :=  …
                               x✝))
                     -/
  map_mul' _ _ := by ext; rfl
                          /-
                            🎉 no goals
                          -/


theorem fixingSubgroup_fixedField [FiniteDimensional F E] : fixingSubgroup (fixedField H) = H := by
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    H : Subgroup (AlgEquiv F E E)
    inst✝ : FiniteDimensional F E
    ⊢ Eq (IntermediateField.fixedField H).fixingSubgroup H
  -/
  have H_le : H ≤ fixingSubgroup (fixedField H) := (le_iff_le _ _).mp le_rfl
  classical
  suffices Fintype.card H = Fintype.card (fixingSubgroup (fixedField H)) by
    exact SetLike.coe_injective (Set.eq_of_inclusion_surjective
      ((Fintype.bijective_iff_injective_and_card (Set.inclusion H_le)).mpr
        ⟨Set.inclusion_injective H_le, this⟩).2).symm
  apply Fintype.card_congr
  refine (FixedPoints.toAlgHomEquiv H E).trans ?_
  refine (algEquivEquivAlgHom (fixedField H) E).toEquiv.symm.trans ?_
  exact (fixingSubgroupEquiv (fixedField H)).toEquiv.symm

-- Porting note: added `fixedField.smul` for `fixedField.isScalarTower`

instance fixedField.smul : SMul K (fixedField (fixingSubgroup K)) where
  smul x y := ⟨x * y, fun ϕ => by
    /-
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      H : Subgroup (AlgEquiv F E E)
      K : IntermediateField F E
      x : Subtype fun x => Membership.mem K x
      y : Subtype fun x => Membership.mem (IntermediateField.fixedField K.fixingSubg …
      ϕ : Subtype fun x => Membership.mem K.fixingSubgroup x
      ⊢ Eq (HSMul.hSMul ϕ (HMul.hMul ↑x ↑y)) (HMul.hMul ↑x ↑y)
    -/
    rw [smul_mul', show ϕ • (x : E) = ↑x from ϕ.2 x, show ϕ • (y : E) = ↑y from y.2 ϕ]⟩
    /-
      🎉 no goals
    -/


instance fixedField.algebra : Algebra K (fixedField (fixingSubgroup K)) where
  toFun x := ⟨x, fun ϕ => Subtype.mem ϕ x⟩
  map_zero' := rfl
  map_add' _ _ := rfl
  map_one' := rfl
  map_mul' _ _ := rfl
  commutes' _ _ := mul_comm _ _
  smul_def' _ _ := rfl


instance fixedField.isScalarTower : IsScalarTower K (fixedField (fixingSubgroup K)) E :=
  ⟨fun _ _ _ => mul_assoc _ _ _⟩


theorem fixedField_fixingSubgroup [FiniteDimensional F E] [h : IsGalois F E] :
    IntermediateField.fixedField (IntermediateField.fixingSubgroup K) = K := by
  have K_le : K ≤ IntermediateField.fixedField (IntermediateField.fixingSubgroup K) :=
    (IntermediateField.le_iff_le _ _).mpr le_rfl
  suffices
    finrank K E = finrank (IntermediateField.fixedField (IntermediateField.fixingSubgroup K)) E by
    exact (IntermediateField.eq_of_le_of_finrank_eq' K_le this).symm
  classical
  rw [IntermediateField.finrank_fixedField_eq_card,
    Fintype.card_congr (IntermediateField.fixingSubgroupEquiv K).toEquiv]
  exact (card_aut_eq_finrank K E).symm


theorem card_fixingSubgroup_eq_finrank [DecidablePred (· ∈ IntermediateField.fixingSubgroup K)]
    [FiniteDimensional F E] [IsGalois F E] :
    Fintype.card (IntermediateField.fixingSubgroup K) = finrank K E := by
  /-
    F : Type u_1
    inst✝⁵ : Field F
    E : Type u_2
    inst✝⁴ : Field E
    inst✝³ : Algebra F E
    K : IntermediateField F E
    inst✝² : DecidablePred fun x => Membership.mem K.fixingSubgroup x
    inst✝¹ : FiniteDimensional F E
    inst✝ : IsGalois F E
    ⊢ Eq (Fintype.card (Subtype fun x => Membership.mem K.fixingSubgroup x)) (Modu …
  -/
  conv_rhs => rw [← fixedField_fixingSubgroup K, IntermediateField.finrank_fixedField_eq_card]
  /-
    🎉 no goals
  -/


/-- The Galois correspondence from intermediate fields to subgroups. -/
@[stacks 09DW]
def intermediateFieldEquivSubgroup [FiniteDimensional F E] [IsGalois F E] :
    IntermediateField F E ≃o (Subgroup (E ≃ₐ[F] E))ᵒᵈ where
  toFun := IntermediateField.fixingSubgroup
  invFun := IntermediateField.fixedField
  left_inv K := fixedField_fixingSubgroup K
  right_inv H := IntermediateField.fixingSubgroup_fixedField H
  map_rel_iff' {K L} := by
    /-
      F : Type u_1
      inst✝⁴ : Field F
      E : Type u_2
      inst✝³ : Field E
      inst✝² : Algebra F E
      H : Subgroup (AlgEquiv F E E)
      K✝ : IntermediateField F E
      inst✝¹ : FiniteDimensional F E
      inst✝ : IsGalois F E
      K L : IntermediateField F E
      ⊢ Iff (LE.le ({ toFun := IntermediateField.fixingSubgroup, invFun := Intermedi …
    -/
    rw [← fixedField_fixingSubgroup L, IntermediateField.le_iff_le, fixedField_fixingSubgroup L]
    /-
      F : Type u_1
      inst✝⁴ : Field F
      E : Type u_2
      inst✝³ : Field E
      inst✝² : Algebra F E
      H : Subgroup (AlgEquiv F E E)
      K✝ : IntermediateField F E
      inst✝¹ : FiniteDimensional F E
      inst✝ : IsGalois F E
      K L : IntermediateField F E
      ⊢ Iff (LE.le ({ toFun := IntermediateField.fixingSubgroup, invFun := Intermedi …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The Galois correspondence as a `GaloisInsertion` -/
def galoisInsertionIntermediateFieldSubgroup [FiniteDimensional F E] :
    GaloisInsertion (OrderDual.toDual ∘
      (IntermediateField.fixingSubgroup : IntermediateField F E → Subgroup (E ≃ₐ[F] E)))
      ((IntermediateField.fixedField : Subgroup (E ≃ₐ[F] E) → IntermediateField F E) ∘
        OrderDual.toDual) where
  choice K _ := IntermediateField.fixingSubgroup K
  gc K H := (IntermediateField.le_iff_le H K).symm
  le_l_u H := le_of_eq (IntermediateField.fixingSubgroup_fixedField H).symm
  choice_eq _ _ := rfl


/-- The Galois correspondence as a `GaloisCoinsertion` -/
def galoisCoinsertionIntermediateFieldSubgroup [FiniteDimensional F E] [IsGalois F E] :
    GaloisCoinsertion (OrderDual.toDual ∘
      (IntermediateField.fixingSubgroup : IntermediateField F E → Subgroup (E ≃ₐ[F] E)))
      ((IntermediateField.fixedField : Subgroup (E ≃ₐ[F] E) → IntermediateField F E) ∘
        OrderDual.toDual) where
  choice H _ := IntermediateField.fixedField H
  gc K H := (IntermediateField.le_iff_le H K).symm
  u_l_le K := le_of_eq (fixedField_fixingSubgroup K)
  choice_eq _ _ := rfl


lemma IntermediateField.restrictNormalHom_ker (E : IntermediateField K L) [Normal K E] :
    (restrictNormalHom E).ker = E.fixingSubgroup := by
  simp [fixingSubgroup, Subgroup.ext_iff, AlgEquiv.ext_iff, Subtype.ext_iff,
    restrictNormalHom_apply, mem_fixingSubgroup_iff]


/-- If `H` is a normal Subgroup of `Gal(L / K)`, then `fixedField H` is Galois over `K`. -/
instance of_fixedField_normal_subgroup [IsGalois K L]
    (H : Subgroup (L ≃ₐ[K] L)) [hn : Subgroup.Normal H] : IsGalois K (fixedField H) where
  to_isSeparable := Algebra.isSeparable_tower_bot_of_isSeparable K (fixedField H) L
  to_normal := by
    /-
      F : Type u_1
      inst✝⁶ : Field F
      E✝ : Type u_2
      inst✝⁵ : Field E✝
      inst✝⁴ : Algebra F E✝
      H✝ : Subgroup (AlgEquiv F E✝ E✝)
      K✝ : IntermediateField F E✝
      K : Type u_3
      L : Type u_4
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      E : IntermediateField K L
      inst✝ : IsGalois K L
      H : Subgroup (AlgEquiv K L L)
      hn : H.Normal
      ⊢ Normal K (Subtype fun x => Membership.mem (IntermediateField.fixedField H) x)
    -/
    apply normal_iff_forall_map_le'.mpr
    /-
      F : Type u_1
      inst✝⁶ : Field F
      E✝ : Type u_2
      inst✝⁵ : Field E✝
      inst✝⁴ : Algebra F E✝
      H✝ : Subgroup (AlgEquiv F E✝ E✝)
      K✝ : IntermediateField F E✝
      K : Type u_3
      L : Type u_4
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      E : IntermediateField K L
      inst✝ : IsGalois K L
      H : Subgroup (AlgEquiv K L L)
      hn : H.Normal
      ⊢ ∀ (σ : AlgEquiv K L L), LE.le (IntermediateField.map (↑σ) (IntermediateField …
    -/
    rintro σ x ⟨a, ha, rfl⟩ τ
    /-
      case intro.intro
      F : Type u_1
      inst✝⁶ : Field F
      E✝ : Type u_2
      inst✝⁵ : Field E✝
      inst✝⁴ : Algebra F E✝
      H✝ : Subgroup (AlgEquiv F E✝ E✝)
      K✝ : IntermediateField F E✝
      K : Type u_3
      L : Type u_4
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      E : IntermediateField K L
      inst✝ : IsGalois K L
      H : Subgroup (AlgEquiv K L L)
      hn : H.Normal
      σ : AlgEquiv K L L
      a : L
      ha : Membership.mem (↑(IntermediateField.fixedField H).toSubsemiring) a
      τ : Subtype fun x => Membership.mem H x
      ⊢ Eq (HSMul.hSMul τ (↑↑σ a)) (↑↑σ a)
    -/
    exact (symm_apply_eq σ).mp (ha ⟨σ⁻¹ * τ * σ, Subgroup.Normal.conj_mem' hn τ.1 τ.2 σ⟩)
    /-
      🎉 no goals
    -/


/-- If `H` is a normal Subgroup of `Gal(L / K)`, then `Gal(fixedField H / K)` is isomorphic to
`Gal(L / K) ⧸ H`. -/
noncomputable def normalAutEquivQuotient [FiniteDimensional K L] [IsGalois K L]
    (H : Subgroup (L ≃ₐ[K] L)) [Subgroup.Normal H] :
    (L ≃ₐ[K] L) ⧸ H ≃* ((fixedField H) ≃ₐ[K] (fixedField H)) :=
  (QuotientGroup.quotientMulEquivOfEq ((fixingSubgroup_fixedField H).symm.trans
  (fixedField H).restrictNormalHom_ker.symm)).trans <|
  QuotientGroup.quotientKerEquivOfSurjective (restrictNormalHom (fixedField H)) <|
  restrictNormalHom_surjective L


lemma normalAutEquivQuotient_apply [FiniteDimensional K L] [IsGalois K L]
    (H : Subgroup (L ≃ₐ[K] L)) [Subgroup.Normal H] (σ : (L ≃ₐ[K] L)) :
    normalAutEquivQuotient H σ = (restrictNormalHom (fixedField H)) σ := rfl


@[simp]
theorem map_fixingSubgroup (σ : L ≃ₐ[K] L) :
    (E.map σ).fixingSubgroup = (MulAut.conj σ) • E.fixingSubgroup := by
  /-
    K : Type u_3
    L : Type u_4
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    E : IntermediateField K L
    σ : AlgEquiv K L L
    ⊢ Eq (IntermediateField.map (↑σ) E).fixingSubgroup (HSMul.hSMul (MulAut.conj σ …
  -/
  ext τ
  simp only [coe_map, AlgHom.coe_coe, Set.mem_image, SetLike.mem_coe, AlgEquiv.smul_def,
    forall_exists_index, and_imp, forall_apply_eq_imp_iff₂, Subtype.forall,
    Subgroup.mem_pointwise_smul_iff_inv_smul_mem, ← symm_apply_eq,
    IntermediateField.fixingSubgroup, mem_fixingSubgroup_iff]
  /-
    case h
    K : Type u_3
    L : Type u_4
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    E : IntermediateField K L
    σ τ : AlgEquiv K L L
    ⊢ Iff (∀ (a : L), Membership.mem E a → Eq (σ.symm (τ (σ a))) a) (∀ (y : L), Me …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Let `E` be an intermediateField of a Galois extension `L / K`. If `E / K` is
Galois extension, then `E.fixingSubgroup` is a normal subgroup of `Gal(L / K)`. -/
instance fixingSubgroup_normal_of_isGalois [IsGalois K L] [IsGalois K E] :
    E.fixingSubgroup.Normal := by
  /-
    F : Type u_1
    inst✝⁷ : Field F
    E✝ : Type u_2
    inst✝⁶ : Field E✝
    inst✝⁵ : Algebra F E✝
    H : Subgroup (AlgEquiv F E✝ E✝)
    K✝ : IntermediateField F E✝
    K : Type u_3
    L : Type u_4
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    E : IntermediateField K L
    inst✝¹ : IsGalois K L
    inst✝ : IsGalois K (Subtype fun x => Membership.mem E x)
    ⊢ E.fixingSubgroup.Normal
  -/
  apply Subgroup.Normal.of_conjugate_fixed (fun σ ↦ ?_)
  /-
    F : Type u_1
    inst✝⁷ : Field F
    E✝ : Type u_2
    inst✝⁶ : Field E✝
    inst✝⁵ : Algebra F E✝
    H : Subgroup (AlgEquiv F E✝ E✝)
    K✝ : IntermediateField F E✝
    K : Type u_3
    L : Type u_4
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    E : IntermediateField K L
    inst✝¹ : IsGalois K L
    inst✝ : IsGalois K (Subtype fun x => Membership.mem E x)
    σ : AlgEquiv K L L
    ⊢ Eq (HSMul.hSMul (MulAut.conj σ) E.fixingSubgroup) E.fixingSubgroup
  -/
  rw [← map_fixingSubgroup, normal_iff_forall_map_eq'.mp inferInstance σ]
  /-
    🎉 no goals
  -/


theorem is_separable_splitting_field [FiniteDimensional F E] [IsGalois F E] :
    ∃ p : F[X], p.Separable ∧ p.IsSplittingField F E := by
  /-
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    inst✝¹ : FiniteDimensional F E
    inst✝ : IsGalois F E
    ⊢ Exists fun p => And p.Separable (Polynomial.IsSplittingField F E p)
  -/
  cases' Field.exists_primitive_element F E with α h1
  /-
    case intro
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    inst✝¹ : FiniteDimensional F E
    inst✝ : IsGalois F E
    α : E
    h1 : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    ⊢ Exists fun p => And p.Separable (Polynomial.IsSplittingField F E p)
  -/
  use minpoly F α, separable F α, IsGalois.splits F α
  /-
    case adjoin_rootSet'
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    inst✝¹ : FiniteDimensional F E
    inst✝ : IsGalois F E
    α : E
    h1 : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    ⊢ Eq (Algebra.adjoin F ((minpoly F α).rootSet E)) Top.top
  -/
  rw [eq_top_iff, ← IntermediateField.top_toSubalgebra, ← h1]
  /-
    case adjoin_rootSet'
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    inst✝¹ : FiniteDimensional F E
    inst✝ : IsGalois F E
    α : E
    h1 : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    ⊢ LE.le (IntermediateField.adjoin F (Singleton.singleton α)).toSubalgebra (Alg …
  -/
  rw [IntermediateField.adjoin_simple_toSubalgebra_of_integral (integral F α)]
  /-
    case adjoin_rootSet'
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    inst✝¹ : FiniteDimensional F E
    inst✝ : IsGalois F E
    α : E
    h1 : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    ⊢ LE.le (Algebra.adjoin F (Singleton.singleton α)) (Algebra.adjoin F ((minpoly …
  -/
  apply Algebra.adjoin_mono
  /-
    case adjoin_rootSet'.H
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    inst✝¹ : FiniteDimensional F E
    inst✝ : IsGalois F E
    α : E
    h1 : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    ⊢ HasSubset.Subset (Singleton.singleton α) ((minpoly F α).rootSet E)
  -/
  rw [Set.singleton_subset_iff, Polynomial.mem_rootSet]
  /-
    case adjoin_rootSet'.H
    F : Type u_1
    inst✝⁴ : Field F
    E : Type u_2
    inst✝³ : Field E
    inst✝² : Algebra F E
    inst✝¹ : FiniteDimensional F E
    inst✝ : IsGalois F E
    α : E
    h1 : Eq (IntermediateField.adjoin F (Singleton.singleton α)) Top.top
    ⊢ And (Ne (minpoly F α) 0) (Eq ((Polynomial.aeval α) (minpoly F α)) 0)
  -/
  exact ⟨minpoly.ne_zero (integral F α), minpoly.aeval _ _⟩
  /-
    🎉 no goals
  -/


theorem of_fixedField_eq_bot [FiniteDimensional F E]
    (h : IntermediateField.fixedField (⊤ : Subgroup (E ≃ₐ[F] E)) = ⊥) : IsGalois F E := by
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    h : Eq (IntermediateField.fixedField Top.top) Bot.bot
    ⊢ IsGalois F E
  -/
  rw [← isGalois_iff_isGalois_bot, ← h]
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    h : Eq (IntermediateField.fixedField Top.top) Bot.bot
    ⊢ IsGalois (Subtype fun x => Membership.mem (IntermediateField.fixedField Top. …
  -/
  classical exact IsGalois.of_fixed_field E (⊤ : Subgroup (E ≃ₐ[F] E))
  /-
    🎉 no goals
  -/


/-- Let $E / F$ be a finite extension of fields. If $|\text{Aut}(E/F)| = [E : F]$, then
$E$ is Galois over $F$. -/
@[stacks 09I1 "'if' part"]
theorem of_card_aut_eq_finrank [FiniteDimensional F E]
    (h : Fintype.card (E ≃ₐ[F] E) = finrank F E) : IsGalois F E := by
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    h : Eq (Fintype.card (AlgEquiv F E E)) (Module.finrank F E)
    ⊢ IsGalois F E
  -/
  apply of_fixedField_eq_bot
  /-
    case h
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    h : Eq (Fintype.card (AlgEquiv F E E)) (Module.finrank F E)
    ⊢ Eq (IntermediateField.fixedField Top.top) Bot.bot
  -/
  have p : 0 < finrank (IntermediateField.fixedField (⊤ : Subgroup (E ≃ₐ[F] E))) E := finrank_pos
  classical
  rw [← IntermediateField.finrank_eq_one_iff, ← mul_left_inj' (ne_of_lt p).symm,
    finrank_mul_finrank, ← h, one_mul, IntermediateField.finrank_fixedField_eq_card]
  apply Fintype.card_congr
  exact
    { toFun := fun g => ⟨g, Subgroup.mem_top g⟩
      invFun := (↑)
      left_inv := fun g => rfl
      right_inv := fun _ => by ext; rfl }


theorem of_separable_splitting_field_aux [hFE : FiniteDimensional F E] [sp : p.IsSplittingField F E]
    (hp : p.Separable) (K : Type*) [Field K] [Algebra F K] [Algebra K E] [IsScalarTower F K E]
    {x : E} (hx : x ∈ p.aroots E)
    -- these are both implied by `hFE`, but as they carry data this makes the lemma more general
    [Fintype (K →ₐ[F] E)]
    [Fintype (K⟮x⟯.restrictScalars F →ₐ[F] E)] :
    Fintype.card (K⟮x⟯.restrictScalars F →ₐ[F] E) = Fintype.card (K →ₐ[F] E) * finrank K K⟮x⟯ := by
  /-
    F : Type u_1
    inst✝⁸ : Field F
    E : Type u_2
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    p : Polynomial F
    hFE : FiniteDimensional F E
    sp : Polynomial.IsSplittingField F E p
    hp : p.Separable
    K : Type u_3
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra K E
    inst✝² : IsScalarTower F K E
    x : E
    hx : Membership.mem (p.aroots E) x
    inst✝¹ : Fintype (AlgHom F K E)
    inst✝ : Fintype (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFiel …
    ⊢ Eq (Fintype.card (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateF …
  -/
  have h : IsIntegral K x := (isIntegral_of_noetherian (IsNoetherian.iff_fg.2 hFE) x).tower_top
  have h1 : p ≠ 0 := fun hp => by
    rw [hp, Polynomial.aroots_zero] at hx
    exact Multiset.not_mem_zero x hx
  have h2 : minpoly K x ∣ p.map (algebraMap F K) := by
    apply minpoly.dvd
    rw [Polynomial.aeval_def, Polynomial.eval₂_map, ← Polynomial.eval_map, ←
      IsScalarTower.algebraMap_eq]
    exact (Polynomial.mem_roots (Polynomial.map_ne_zero h1)).mp hx
  let key_equiv : (K⟮x⟯.restrictScalars F →ₐ[F] E) ≃
      Σ f : K →ₐ[F] E, @AlgHom K K⟮x⟯ E _ _ _ _ (RingHom.toAlgebra f) := by
    change (K⟮x⟯ →ₐ[F] E) ≃ Σ f : K →ₐ[F] E, _
    exact algHomEquivSigma
  haveI : ∀ f : K →ₐ[F] E, Fintype (@AlgHom K K⟮x⟯ E _ _ _ _ (RingHom.toAlgebra f)) := fun f => by
    have := Fintype.ofEquiv _ key_equiv
    apply Fintype.ofInjective (Sigma.mk f) fun _ _ H => eq_of_heq (Sigma.ext_iff.mp H).2
  /-
    F : Type u_1
    inst✝⁸ : Field F
    E : Type u_2
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    p : Polynomial F
    hFE : FiniteDimensional F E
    sp : Polynomial.IsSplittingField F E p
    hp : p.Separable
    K : Type u_3
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra K E
    inst✝² : IsScalarTower F K E
    x : E
    hx : Membership.mem (p.aroots E) x
    inst✝¹ : Fintype (AlgHom F K E)
    inst✝ : Fintype (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFiel …
    h : IsIntegral K x
    h1 : Ne p 0
    h2 : Dvd.dvd (minpoly K x) (Polynomial.map (algebraMap F K) p)
    key_equiv : Equiv (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFi …
    this : (f : AlgHom F K E) → Fintype (AlgHom K (Subtype fun x_1 => Membership.m …
    ⊢ Eq (Fintype.card (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateF …
  -/
  rw [Fintype.card_congr key_equiv, Fintype.card_sigma, IntermediateField.adjoin.finrank h]
  /-
    F : Type u_1
    inst✝⁸ : Field F
    E : Type u_2
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    p : Polynomial F
    hFE : FiniteDimensional F E
    sp : Polynomial.IsSplittingField F E p
    hp : p.Separable
    K : Type u_3
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra K E
    inst✝² : IsScalarTower F K E
    x : E
    hx : Membership.mem (p.aroots E) x
    inst✝¹ : Fintype (AlgHom F K E)
    inst✝ : Fintype (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFiel …
    h : IsIntegral K x
    h1 : Ne p 0
    h2 : Dvd.dvd (minpoly K x) (Polynomial.map (algebraMap F K) p)
    key_equiv : Equiv (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFi …
    this : (f : AlgHom F K E) → Fintype (AlgHom K (Subtype fun x_1 => Membership.m …
    ⊢ Eq (Finset.univ.sum fun i => Fintype.card (AlgHom K (Subtype fun x_1 => Memb …
  -/
  apply Finset.sum_const_nat
  /-
    case h₁
    F : Type u_1
    inst✝⁸ : Field F
    E : Type u_2
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    p : Polynomial F
    hFE : FiniteDimensional F E
    sp : Polynomial.IsSplittingField F E p
    hp : p.Separable
    K : Type u_3
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra K E
    inst✝² : IsScalarTower F K E
    x : E
    hx : Membership.mem (p.aroots E) x
    inst✝¹ : Fintype (AlgHom F K E)
    inst✝ : Fintype (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFiel …
    h : IsIntegral K x
    h1 : Ne p 0
    h2 : Dvd.dvd (minpoly K x) (Polynomial.map (algebraMap F K) p)
    key_equiv : Equiv (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFi …
    this : (f : AlgHom F K E) → Fintype (AlgHom K (Subtype fun x_1 => Membership.m …
    ⊢ ∀ (x_1 : AlgHom F K E), Membership.mem Finset.univ x_1 → Eq (Fintype.card (A …
  -/
  intro f _
  /-
    case h₁
    F : Type u_1
    inst✝⁸ : Field F
    E : Type u_2
    inst✝⁷ : Field E
    inst✝⁶ : Algebra F E
    p : Polynomial F
    hFE : FiniteDimensional F E
    sp : Polynomial.IsSplittingField F E p
    hp : p.Separable
    K : Type u_3
    inst✝⁵ : Field K
    inst✝⁴ : Algebra F K
    inst✝³ : Algebra K E
    inst✝² : IsScalarTower F K E
    x : E
    hx : Membership.mem (p.aroots E) x
    inst✝¹ : Fintype (AlgHom F K E)
    inst✝ : Fintype (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFiel …
    h : IsIntegral K x
    h1 : Ne p 0
    h2 : Dvd.dvd (minpoly K x) (Polynomial.map (algebraMap F K) p)
    key_equiv : Equiv (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFi …
    this : (f : AlgHom F K E) → Fintype (AlgHom K (Subtype fun x_1 => Membership.m …
    f : AlgHom F K E
    a✝ : Membership.mem Finset.univ f
    ⊢ Eq (Fintype.card (AlgHom K (Subtype fun x_1 => Membership.mem (IntermediateF …
  -/
  rw [← @IntermediateField.card_algHom_adjoin_integral K _ E _ _ x E _ (RingHom.toAlgebra f) h]
    /-
      case h₁
      F : Type u_1
      inst✝⁸ : Field F
      E : Type u_2
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      p : Polynomial F
      hFE : FiniteDimensional F E
      sp : Polynomial.IsSplittingField F E p
      hp : p.Separable
      K : Type u_3
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra K E
      inst✝² : IsScalarTower F K E
      x : E
      hx : Membership.mem (p.aroots E) x
      inst✝¹ : Fintype (AlgHom F K E)
      inst✝ : Fintype (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFiel …
      h : IsIntegral K x
      h1 : Ne p 0
      h2 : Dvd.dvd (minpoly K x) (Polynomial.map (algebraMap F K) p)
      key_equiv : Equiv (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFi …
      this : (f : AlgHom F K E) → Fintype (AlgHom K (Subtype fun x_1 => Membership.m …
      f : AlgHom F K E
      a✝ : Membership.mem Finset.univ f
      ⊢ Eq (Fintype.card (AlgHom K (Subtype fun x_1 => Membership.mem (IntermediateF …
    -/
  · congr!
    /-
      🎉 no goals
    -/
    /-
      case h₁.h_sep
      F : Type u_1
      inst✝⁸ : Field F
      E : Type u_2
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      p : Polynomial F
      hFE : FiniteDimensional F E
      sp : Polynomial.IsSplittingField F E p
      hp : p.Separable
      K : Type u_3
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra K E
      inst✝² : IsScalarTower F K E
      x : E
      hx : Membership.mem (p.aroots E) x
      inst✝¹ : Fintype (AlgHom F K E)
      inst✝ : Fintype (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFiel …
      h : IsIntegral K x
      h1 : Ne p 0
      h2 : Dvd.dvd (minpoly K x) (Polynomial.map (algebraMap F K) p)
      key_equiv : Equiv (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFi …
      this : (f : AlgHom F K E) → Fintype (AlgHom K (Subtype fun x_1 => Membership.m …
      f : AlgHom F K E
      a✝ : Membership.mem Finset.univ f
      ⊢ IsSeparable K x
    -/
  · exact Polynomial.Separable.of_dvd ((Polynomial.separable_map (algebraMap F K)).mpr hp) h2
    /-
      🎉 no goals
    -/
    /-
      case h₁.h_splits
      F : Type u_1
      inst✝⁸ : Field F
      E : Type u_2
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      p : Polynomial F
      hFE : FiniteDimensional F E
      sp : Polynomial.IsSplittingField F E p
      hp : p.Separable
      K : Type u_3
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra K E
      inst✝² : IsScalarTower F K E
      x : E
      hx : Membership.mem (p.aroots E) x
      inst✝¹ : Fintype (AlgHom F K E)
      inst✝ : Fintype (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFiel …
      h : IsIntegral K x
      h1 : Ne p 0
      h2 : Dvd.dvd (minpoly K x) (Polynomial.map (algebraMap F K) p)
      key_equiv : Equiv (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFi …
      this : (f : AlgHom F K E) → Fintype (AlgHom K (Subtype fun x_1 => Membership.m …
      f : AlgHom F K E
      a✝ : Membership.mem Finset.univ f
      ⊢ Polynomial.Splits (algebraMap K E) (minpoly K x)
    -/
  · refine Polynomial.splits_of_splits_of_dvd _ (Polynomial.map_ne_zero h1) ?_ h2
    -- Porting note: use unification instead of synthesis for one argument of `algebraMap_eq`
    /-
      case h₁.h_splits
      F : Type u_1
      inst✝⁸ : Field F
      E : Type u_2
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      p : Polynomial F
      hFE : FiniteDimensional F E
      sp : Polynomial.IsSplittingField F E p
      hp : p.Separable
      K : Type u_3
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra K E
      inst✝² : IsScalarTower F K E
      x : E
      hx : Membership.mem (p.aroots E) x
      inst✝¹ : Fintype (AlgHom F K E)
      inst✝ : Fintype (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFiel …
      h : IsIntegral K x
      h1 : Ne p 0
      h2 : Dvd.dvd (minpoly K x) (Polynomial.map (algebraMap F K) p)
      key_equiv : Equiv (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFi …
      this : (f : AlgHom F K E) → Fintype (AlgHom K (Subtype fun x_1 => Membership.m …
      f : AlgHom F K E
      a✝ : Membership.mem Finset.univ f
      ⊢ Polynomial.Splits (algebraMap K E) (Polynomial.map (algebraMap F K) p)
    -/
    rw [Polynomial.splits_map_iff, ← @IsScalarTower.algebraMap_eq _ _ _ _ _ _ _ (_) _ _]
    /-
      case h₁.h_splits
      F : Type u_1
      inst✝⁸ : Field F
      E : Type u_2
      inst✝⁷ : Field E
      inst✝⁶ : Algebra F E
      p : Polynomial F
      hFE : FiniteDimensional F E
      sp : Polynomial.IsSplittingField F E p
      hp : p.Separable
      K : Type u_3
      inst✝⁵ : Field K
      inst✝⁴ : Algebra F K
      inst✝³ : Algebra K E
      inst✝² : IsScalarTower F K E
      x : E
      hx : Membership.mem (p.aroots E) x
      inst✝¹ : Fintype (AlgHom F K E)
      inst✝ : Fintype (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFiel …
      h : IsIntegral K x
      h1 : Ne p 0
      h2 : Dvd.dvd (minpoly K x) (Polynomial.map (algebraMap F K) p)
      key_equiv : Equiv (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateFi …
      this : (f : AlgHom F K E) → Fintype (AlgHom K (Subtype fun x_1 => Membership.m …
      f : AlgHom F K E
      a✝ : Membership.mem Finset.univ f
      ⊢ Polynomial.Splits (algebraMap F E) p
    -/
    exact sp.splits
    /-
      🎉 no goals
    -/


theorem of_separable_splitting_field [sp : p.IsSplittingField F E] (hp : p.Separable) :
    IsGalois F E := by
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    sp : Polynomial.IsSplittingField F E p
    hp : p.Separable
    ⊢ IsGalois F E
  -/
  haveI hFE : FiniteDimensional F E := Polynomial.IsSplittingField.finiteDimensional E p
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    sp : Polynomial.IsSplittingField F E p
    hp : p.Separable
    hFE : FiniteDimensional F E
    ⊢ IsGalois F E
  -/
  letI := Classical.decEq E
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    sp : Polynomial.IsSplittingField F E p
    hp : p.Separable
    hFE : FiniteDimensional F E
    this : DecidableEq E := Classical.decEq E
    ⊢ IsGalois F E
  -/
  let s := p.rootSet E
  have adjoin_root : IntermediateField.adjoin F s = ⊤ := by
    apply IntermediateField.toSubalgebra_injective
    rw [IntermediateField.top_toSubalgebra, ← top_le_iff, ← sp.adjoin_rootSet]
    apply IntermediateField.algebra_adjoin_le_adjoin
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    sp : Polynomial.IsSplittingField F E p
    hp : p.Separable
    hFE : FiniteDimensional F E
    this : DecidableEq E := Classical.decEq E
    s : Set E := p.rootSet E
    adjoin_root : Eq (IntermediateField.adjoin F s) Top.top
    ⊢ IsGalois F E
  -/
  let P : IntermediateField F E → Prop := fun K => Fintype.card (K →ₐ[F] E) = finrank F K
  suffices P (IntermediateField.adjoin F s) by
    rw [adjoin_root] at this
    apply of_card_aut_eq_finrank
    rw [← Eq.trans this (LinearEquiv.finrank_eq IntermediateField.topEquiv.toLinearEquiv)]
    exact Fintype.card_congr ((algEquivEquivAlgHom F E).toEquiv.trans
      (IntermediateField.topEquiv.symm.arrowCongr AlgEquiv.refl))
  /-
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    sp : Polynomial.IsSplittingField F E p
    hp : p.Separable
    hFE : FiniteDimensional F E
    this : DecidableEq E := Classical.decEq E
    s : Set E := p.rootSet E
    adjoin_root : Eq (IntermediateField.adjoin F s) Top.top
    P : IntermediateField F E → Prop := fun K => Eq (Fintype.card (AlgHom F (Subty …
    ⊢ P (IntermediateField.adjoin F s)
  -/
  apply IntermediateField.induction_on_adjoin_finset _ P
  · have key := IntermediateField.card_algHom_adjoin_integral F (K := E)
      (show IsIntegral F (0 : E) from isIntegral_zero)
    /-
      case base
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      p : Polynomial F
      sp : Polynomial.IsSplittingField F E p
      hp : p.Separable
      hFE : FiniteDimensional F E
      this : DecidableEq E := Classical.decEq E
      s : Set E := p.rootSet E
      adjoin_root : Eq (IntermediateField.adjoin F s) Top.top
      P : IntermediateField F E → Prop := fun K => Eq (Fintype.card (AlgHom F (Subty …
      key : IsSeparable F 0 → Polynomial.Splits (algebraMap F E) (minpoly F 0) → Eq  …
      ⊢ P Bot.bot
    -/
    rw [IsSeparable, minpoly.zero, Polynomial.natDegree_X] at key
    /-
      case base
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      p : Polynomial F
      sp : Polynomial.IsSplittingField F E p
      hp : p.Separable
      hFE : FiniteDimensional F E
      this : DecidableEq E := Classical.decEq E
      s : Set E := p.rootSet E
      adjoin_root : Eq (IntermediateField.adjoin F s) Top.top
      P : IntermediateField F E → Prop := fun K => Eq (Fintype.card (AlgHom F (Subty …
      key : Polynomial.X.Separable → Polynomial.Splits (algebraMap F E) Polynomial.X …
      ⊢ P Bot.bot
    -/
    specialize key Polynomial.separable_X (Polynomial.splits_X (algebraMap F E))
    /-
      case base
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      p : Polynomial F
      sp : Polynomial.IsSplittingField F E p
      hp : p.Separable
      hFE : FiniteDimensional F E
      this : DecidableEq E := Classical.decEq E
      s : Set E := p.rootSet E
      adjoin_root : Eq (IntermediateField.adjoin F s) Top.top
      P : IntermediateField F E → Prop := fun K => Eq (Fintype.card (AlgHom F (Subty …
      key : Eq (Fintype.card (AlgHom F (Subtype fun x => Membership.mem (Intermediat …
      ⊢ P Bot.bot
    -/
    rw [← @Subalgebra.finrank_bot F E _ _ _, ← IntermediateField.bot_toSubalgebra] at key
    /-
      case base
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      p : Polynomial F
      sp : Polynomial.IsSplittingField F E p
      hp : p.Separable
      hFE : FiniteDimensional F E
      this : DecidableEq E := Classical.decEq E
      s : Set E := p.rootSet E
      adjoin_root : Eq (IntermediateField.adjoin F s) Top.top
      P : IntermediateField F E → Prop := fun K => Eq (Fintype.card (AlgHom F (Subty …
      key : Eq (Fintype.card (AlgHom F (Subtype fun x => Membership.mem (Intermediat …
      ⊢ P Bot.bot
    -/
    refine Eq.trans ?_ key
    -- Porting note: use unification instead of synthesis for one argument of `card_congr`
    /-
      case base
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      p : Polynomial F
      sp : Polynomial.IsSplittingField F E p
      hp : p.Separable
      hFE : FiniteDimensional F E
      this : DecidableEq E := Classical.decEq E
      s : Set E := p.rootSet E
      adjoin_root : Eq (IntermediateField.adjoin F s) Top.top
      P : IntermediateField F E → Prop := fun K => Eq (Fintype.card (AlgHom F (Subty …
      key : Eq (Fintype.card (AlgHom F (Subtype fun x => Membership.mem (Intermediat …
      ⊢ Eq (Fintype.card (AlgHom F (Subtype fun x => Membership.mem Bot.bot x) E)) ( …
    -/
    apply @Fintype.card_congr _ _ _ (_) _
    /-
      F : Type u_1
      inst✝² : Field F
      E : Type u_2
      inst✝¹ : Field E
      inst✝ : Algebra F E
      p : Polynomial F
      sp : Polynomial.IsSplittingField F E p
      hp : p.Separable
      hFE : FiniteDimensional F E
      this : DecidableEq E := Classical.decEq E
      s : Set E := p.rootSet E
      adjoin_root : Eq (IntermediateField.adjoin F s) Top.top
      P : IntermediateField F E → Prop := fun K => Eq (Fintype.card (AlgHom F (Subty …
      key : Eq (Fintype.card (AlgHom F (Subtype fun x => Membership.mem (Intermediat …
      ⊢ Equiv (AlgHom F (Subtype fun x => Membership.mem Bot.bot x) E) (AlgHom F (Su …
    -/
    rw [IntermediateField.adjoin_zero]
    /-
      🎉 no goals
    -/
  /-
    case ih
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    sp : Polynomial.IsSplittingField F E p
    hp : p.Separable
    hFE : FiniteDimensional F E
    this : DecidableEq E := Classical.decEq E
    s : Set E := p.rootSet E
    adjoin_root : Eq (IntermediateField.adjoin F s) Top.top
    P : IntermediateField F E → Prop := fun K => Eq (Fintype.card (AlgHom F (Subty …
    ⊢ ∀ (K : IntermediateField F E) (x : E), Membership.mem (p.aroots E).toFinset  …
  -/
  intro K x hx hK
  /-
    case ih
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    sp : Polynomial.IsSplittingField F E p
    hp : p.Separable
    hFE : FiniteDimensional F E
    this : DecidableEq E := Classical.decEq E
    s : Set E := p.rootSet E
    adjoin_root : Eq (IntermediateField.adjoin F s) Top.top
    P : IntermediateField F E → Prop := fun K => Eq (Fintype.card (AlgHom F (Subty …
    K : IntermediateField F E
    x : E
    hx : Membership.mem (p.aroots E).toFinset x
    hK : P K
    ⊢ P (IntermediateField.restrictScalars F (IntermediateField.adjoin (Subtype fu …
  -/
  simp only [P] at *
  /-
    case ih
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    sp : Polynomial.IsSplittingField F E p
    hp : p.Separable
    hFE : FiniteDimensional F E
    this : DecidableEq E := Classical.decEq E
    s : Set E := p.rootSet E
    adjoin_root : Eq (IntermediateField.adjoin F s) Top.top
    P : IntermediateField F E → Prop := fun K => Eq (Fintype.card (AlgHom F (Subty …
    K : IntermediateField F E
    x : E
    hx : Membership.mem (p.aroots E).toFinset x
    hK : Eq (Fintype.card (AlgHom F (Subtype fun x => Membership.mem K x) E)) (Mod …
    ⊢ Eq (Fintype.card (AlgHom F (Subtype fun x_1 => Membership.mem (IntermediateF …
  -/
  rw [of_separable_splitting_field_aux hp K (Multiset.mem_toFinset.mp hx), hK, finrank_mul_finrank]
  /-
    case ih
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    sp : Polynomial.IsSplittingField F E p
    hp : p.Separable
    hFE : FiniteDimensional F E
    this : DecidableEq E := Classical.decEq E
    s : Set E := p.rootSet E
    adjoin_root : Eq (IntermediateField.adjoin F s) Top.top
    P : IntermediateField F E → Prop := fun K => Eq (Fintype.card (AlgHom F (Subty …
    K : IntermediateField F E
    x : E
    hx : Membership.mem (p.aroots E).toFinset x
    hK : Eq (Fintype.card (AlgHom F (Subtype fun x => Membership.mem K x) E)) (Mod …
    ⊢ Eq (Module.finrank F (Subtype fun x_1 => Membership.mem (IntermediateField.a …
  -/
  symm
  /-
    case ih
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    sp : Polynomial.IsSplittingField F E p
    hp : p.Separable
    hFE : FiniteDimensional F E
    this : DecidableEq E := Classical.decEq E
    s : Set E := p.rootSet E
    adjoin_root : Eq (IntermediateField.adjoin F s) Top.top
    P : IntermediateField F E → Prop := fun K => Eq (Fintype.card (AlgHom F (Subty …
    K : IntermediateField F E
    x : E
    hx : Membership.mem (p.aroots E).toFinset x
    hK : Eq (Fintype.card (AlgHom F (Subtype fun x => Membership.mem K x) E)) (Mod …
    ⊢ Eq (Module.finrank F (Subtype fun x_1 => Membership.mem (IntermediateField.r …
  -/
  refine LinearEquiv.finrank_eq ?_
  /-
    case ih
    F : Type u_1
    inst✝² : Field F
    E : Type u_2
    inst✝¹ : Field E
    inst✝ : Algebra F E
    p : Polynomial F
    sp : Polynomial.IsSplittingField F E p
    hp : p.Separable
    hFE : FiniteDimensional F E
    this : DecidableEq E := Classical.decEq E
    s : Set E := p.rootSet E
    adjoin_root : Eq (IntermediateField.adjoin F s) Top.top
    P : IntermediateField F E → Prop := fun K => Eq (Fintype.card (AlgHom F (Subty …
    K : IntermediateField F E
    x : E
    hx : Membership.mem (p.aroots E).toFinset x
    hK : Eq (Fintype.card (AlgHom F (Subtype fun x => Membership.mem K x) E)) (Mod …
    ⊢ LinearEquiv (RingHom.id F) (Subtype fun x_1 => Membership.mem (IntermediateF …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Equivalent characterizations of a Galois extension of finite degree -/
theorem tfae [FiniteDimensional F E] : List.TFAE [
    IsGalois F E,
    IntermediateField.fixedField (⊤ : Subgroup (E ≃ₐ[F] E)) = ⊥,
    Fintype.card (E ≃ₐ[F] E) = finrank F E,
    ∃ p : F[X], p.Separable ∧ p.IsSplittingField F E] := by
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    ⊢ (List.cons (IsGalois F E) (List.cons (Eq (IntermediateField.fixedField Top.t …
  -/
  tfae_have 1 → 2 := fun h ↦ OrderIso.map_bot (@intermediateFieldEquivSubgroup F _ E _ _ _ h).symm
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    tfae_1_to_2 : IsGalois F E → Eq (IntermediateField.fixedField Top.top) Bot.bot
    ⊢ (List.cons (IsGalois F E) (List.cons (Eq (IntermediateField.fixedField Top.t …
  -/
  tfae_have 1 → 3 := fun _ ↦ card_aut_eq_finrank F E
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    tfae_1_to_2 : IsGalois F E → Eq (IntermediateField.fixedField Top.top) Bot.bot
    tfae_1_to_3 : IsGalois F E → Eq (Fintype.card (AlgEquiv F E E)) (Module.finran …
    ⊢ (List.cons (IsGalois F E) (List.cons (Eq (IntermediateField.fixedField Top.t …
  -/
  tfae_have 1 → 4 := fun _ ↦ is_separable_splitting_field F E
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    tfae_1_to_2 : IsGalois F E → Eq (IntermediateField.fixedField Top.top) Bot.bot
    tfae_1_to_3 : IsGalois F E → Eq (Fintype.card (AlgEquiv F E E)) (Module.finran …
    tfae_1_to_4 : IsGalois F E → Exists fun p => And p.Separable (Polynomial.IsSpl …
    ⊢ (List.cons (IsGalois F E) (List.cons (Eq (IntermediateField.fixedField Top.t …
  -/
  tfae_have 2 → 1 := of_fixedField_eq_bot F E
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    tfae_1_to_2 : IsGalois F E → Eq (IntermediateField.fixedField Top.top) Bot.bot
    tfae_1_to_3 : IsGalois F E → Eq (Fintype.card (AlgEquiv F E E)) (Module.finran …
    tfae_1_to_4 : IsGalois F E → Exists fun p => And p.Separable (Polynomial.IsSpl …
    tfae_2_to_1 : Eq (IntermediateField.fixedField Top.top) Bot.bot → IsGalois F E
    ⊢ (List.cons (IsGalois F E) (List.cons (Eq (IntermediateField.fixedField Top.t …
  -/
  tfae_have 3 → 1 := of_card_aut_eq_finrank F E
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    tfae_1_to_2 : IsGalois F E → Eq (IntermediateField.fixedField Top.top) Bot.bot
    tfae_1_to_3 : IsGalois F E → Eq (Fintype.card (AlgEquiv F E E)) (Module.finran …
    tfae_1_to_4 : IsGalois F E → Exists fun p => And p.Separable (Polynomial.IsSpl …
    tfae_2_to_1 : Eq (IntermediateField.fixedField Top.top) Bot.bot → IsGalois F E
    tfae_3_to_1 : Eq (Fintype.card (AlgEquiv F E E)) (Module.finrank F E) → IsGalo …
    ⊢ (List.cons (IsGalois F E) (List.cons (Eq (IntermediateField.fixedField Top.t …
  -/
  tfae_have 4 → 1 := fun ⟨h, hp1, _⟩ ↦ of_separable_splitting_field hp1
  /-
    F : Type u_1
    inst✝³ : Field F
    E : Type u_2
    inst✝² : Field E
    inst✝¹ : Algebra F E
    inst✝ : FiniteDimensional F E
    tfae_1_to_2 : IsGalois F E → Eq (IntermediateField.fixedField Top.top) Bot.bot
    tfae_1_to_3 : IsGalois F E → Eq (Fintype.card (AlgEquiv F E E)) (Module.finran …
    tfae_1_to_4 : IsGalois F E → Exists fun p => And p.Separable (Polynomial.IsSpl …
    tfae_2_to_1 : Eq (IntermediateField.fixedField Top.top) Bot.bot → IsGalois F E
    tfae_3_to_1 : Eq (Fintype.card (AlgEquiv F E E)) (Module.finrank F E) → IsGalo …
    tfae_4_to_1 : (Exists fun p => And p.Separable (Polynomial.IsSplittingField F  …
    ⊢ (List.cons (IsGalois F E) (List.cons (Eq (IntermediateField.fixedField Top.t …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


/-- Let $F / K / k$ be a tower of field extensions. If $F$ is Galois over $k$,
then the normal closure of $K$ over $k$ in $F$ is Galois over $k$. -/
@[stacks 0EXM]
instance IsGalois.normalClosure : IsGalois k (normalClosure k K F) where
  to_isSeparable := Algebra.isSeparable_tower_bot_of_isSeparable k _ F


instance (priority := 100) IsAlgClosure.isGalois (k K : Type*) [Field k] [Field K] [Algebra k K]
    [IsAlgClosure k K] [CharZero k] : IsGalois k K where


