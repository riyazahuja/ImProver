instance : HasProducts.{v} (Type v) := inferInstance


/-- A restatement of `Types.Limit.lift_π_apply` that uses `Pi.π` and `Pi.lift`. -/
@[simp 1001]
theorem pi_lift_π_apply {β : Type v} [Small.{u} β] (f : β → Type u) {P : Type u}
    (s : ∀ b, P ⟶ f b) (b : β) (x : P) :
    (Pi.π f b : (piObj f) → f b) (@Pi.lift β _ _ f _ P s x) = s b x :=
  congr_fun (limit.lift_π (Fan.mk P s) ⟨b⟩) x


/-- A restatement of `Types.Limit.lift_π_apply` that uses `Pi.π` and `Pi.lift`,
with specialized universes. -/
theorem pi_lift_π_apply' {β : Type v} (f : β → Type v) {P : Type v}
    (s : ∀ b, P ⟶ f b) (b : β) (x : P) :
    (Pi.π f b : (piObj f) → f b) (@Pi.lift β _ _ f _ P s x) = s b x := by
  /-
    β : Type v
    f : β → Type v
    P : Type v
    s : (b : β) → Quiver.Hom P (f b)
    b : β
    x : P
    ⊢ Eq (CategoryTheory.Limits.Pi.π f b (CategoryTheory.Limits.Pi.lift s x)) (s b …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A restatement of `Types.Limit.map_π_apply` that uses `Pi.π` and `Pi.map`. -/
@[simp 1001]
theorem pi_map_π_apply {β : Type v} [Small.{u} β] {f g : β → Type u}
    (α : ∀ j, f j ⟶ g j) (b : β) (x) :
    (Pi.π g b : ∏ᶜ g → g b) (Pi.map α x) = α b ((Pi.π f b : ∏ᶜ f → f b) x) :=
  Limit.map_π_apply.{v, u} _ _ _


/-- A restatement of `Types.Limit.map_π_apply` that uses `Pi.π` and `Pi.map`,
with specialized universes. -/
theorem pi_map_π_apply' {β : Type v} {f g : β → Type v} (α : ∀ j, f j ⟶ g j) (b : β) (x) :
    (Pi.π g b : ∏ᶜ g → g b) (Pi.map α x) = α b ((Pi.π f b : ∏ᶜ f → f b) x) := by
  /-
    β : Type v
    f g : β → Type v
    α : (j : β) → Quiver.Hom (f j) (g j)
    b : β
    x : CategoryTheory.Limits.piObj f
    ⊢ Eq (CategoryTheory.Limits.Pi.π g b (CategoryTheory.Limits.Pi.map α x)) (α b  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The category of types has `PUnit` as a terminal object. -/
def terminalLimitCone : Limits.LimitCone (Functor.empty (Type u)) where
  -- Porting note: tidy was able to fill the structure automatically
  cone :=
    { pt := PUnit
      π := (Functor.uniqueFromEmpty _).hom }
  isLimit :=
    { lift := fun _ _ => PUnit.unit
                         /-
                           x✝ : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (Type u))
                           ⊢ ∀ (j : CategoryTheory.Discrete PEmpty.{?u.3437 + 1}), Eq (CategoryTheory.Cat …
                         -/
      fac := fun _ => by rintro ⟨⟨⟩⟩
                         /-
                           🎉 no goals
                         -/
      uniq := fun _ _ _ => by
        /-
          x✝² : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (Type u))
          x✝¹ : Quiver.Hom x✝².pt { pt := PUnit.{u + 1}, π := ((CategoryTheory.Functor.c …
          x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{?u.3437 + 1}), Eq (CategoryTheory. …
          ⊢ Eq x✝¹ ((fun x x => PUnit.unit) x✝²)
        -/
        funext
        /-
          case h
          x✝³ : CategoryTheory.Limits.Cone (CategoryTheory.Functor.empty (Type u))
          x✝² : Quiver.Hom x✝³.pt { pt := PUnit.{u + 1}, π := ((CategoryTheory.Functor.c …
          x✝¹ : ∀ (j : CategoryTheory.Discrete PEmpty.{?u.3437 + 1}), Eq (CategoryTheory …
          x✝ : x✝³.pt
          ⊢ Eq (x✝² x✝) ((fun x x => PUnit.unit) x✝³ x✝)
        -/
        subsingleton }
        /-
          🎉 no goals
        -/


/-- The terminal object in `Type u` is `PUnit`. -/
noncomputable def terminalIso : ⊤_ Type u ≅ PUnit :=
  limit.isoLimitCone terminalLimitCone.{u, 0}


/-- The terminal object in `Type u` is `PUnit`. -/
noncomputable def isTerminalPunit : IsTerminal (PUnit : Type u) :=
  terminalIsTerminal.ofIso terminalIso

-- Porting note: the following three instances have been added to ease
-- the automation in a definition in `AlgebraicTopology.SimplicialSet`

noncomputable instance : Inhabited (⊤_ (Type u)) :=
  ⟨@terminal.from (Type u) _ _ (ULift (Fin 1)) (ULift.up 0)⟩


instance : Subsingleton (⊤_ (Type u)) := ⟨fun a b =>
  congr_fun (@Subsingleton.elim (_ ⟶ ⊤_ (Type u)) _
    (fun _ => a) (fun _ => b)) (ULift.up (0 : Fin 1))⟩


noncomputable instance : Unique (⊤_ (Type u)) := Unique.mk' _


/-- A type is terminal if and only if it contains exactly one element. -/
noncomputable def isTerminalEquivUnique (X : Type u) : IsTerminal X ≃ Unique X :=
  equivOfSubsingletonOfSubsingleton
    (fun h => ((Iso.toEquiv (terminalIsoIsTerminal h).symm).unique))
    (fun _ => IsTerminal.ofIso terminalIsTerminal (Equiv.toIso (Equiv.ofUnique _ _)))


/-- A type is terminal if and only if it is isomorphic to `PUnit`. -/
noncomputable def isTerminalEquivIsoPUnit (X : Type u) : IsTerminal X ≃ (X ≅ PUnit) := by
  calc
    IsTerminal X ≃ Unique X := isTerminalEquivUnique _
    _ ≃ (X ≃ PUnit.{u + 1}) := uniqueEquivEquivUnique _ _
    _ ≃ (X ≅ PUnit) := equivEquivIso


/-- The category of types has `PEmpty` as an initial object. -/
def initialColimitCocone : Limits.ColimitCocone (Functor.empty (Type u)) where
  -- Porting note: tidy was able to fill the structure automatically
  cocone :=
    { pt := PEmpty
      ι := (Functor.uniqueFromEmpty _).inv }
  isColimit :=
                          /-
                            x✝ : CategoryTheory.Limits.Cocone (CategoryTheory.Functor.empty (Type u))
                            ⊢ Quiver.Hom { pt := PEmpty.{u + 1}, ι := ((CategoryTheory.Functor.const (Cate …
                          -/
    { desc := fun _ => by rintro ⟨⟩
                          /-
                            🎉 no goals
                          -/
                         /-
                           x✝ : CategoryTheory.Limits.Cocone (CategoryTheory.Functor.empty (Type u))
                           ⊢ ∀ (j : CategoryTheory.Discrete PEmpty.{?u.7820 + 1}), Eq (CategoryTheory.Cat …
                         -/
      fac := fun _ => by rintro ⟨⟨⟩⟩
                         /-
                           🎉 no goals
                         -/
                              /-
                                x✝² : CategoryTheory.Limits.Cocone (CategoryTheory.Functor.empty (Type u))
                                x✝¹ : Quiver.Hom { pt := PEmpty.{u + 1}, ι := ((CategoryTheory.Functor.const ( …
                                x✝ : ∀ (j : CategoryTheory.Discrete PEmpty.{?u.7820 + 1}), Eq (CategoryTheory. …
                                ⊢ Eq x✝¹ ((fun x a => PEmpty.casesOn (fun x_1 => x.pt) a) x✝²)
                              -/
      uniq := fun _ _ _ => by funext x; cases x }
                                        /-
                                          🎉 no goals
                                        -/


/-- The initial object in `Type u` is `PEmpty`. -/
noncomputable def initialIso : ⊥_ Type u ≅ PEmpty :=
  colimit.isoColimitCocone initialColimitCocone.{u, 0}


/-- The initial object in `Type u` is `PEmpty`. -/
noncomputable def isInitialPunit : IsInitial (PEmpty : Type u) :=
  initialIsInitial.ofIso initialIso


/-- An object in `Type u` is initial if and only if it is empty. -/
lemma initial_iff_empty (X : Type u) : Nonempty (IsInitial X) ↔ IsEmpty X := by
  /-
    X : Type u
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsInitial X)) (IsEmpty X)
  -/
  constructor
    /-
      case mp
      X : Type u
      ⊢ Nonempty (CategoryTheory.Limits.IsInitial X) → IsEmpty X
    -/
  · intro ⟨h⟩
    /-
      case mp
      X : Type u
      h : CategoryTheory.Limits.IsInitial X
      ⊢ IsEmpty X
    -/
    exact Function.isEmpty (IsInitial.to h PEmpty)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u
      ⊢ IsEmpty X → Nonempty (CategoryTheory.Limits.IsInitial X)
    -/
  · intro h
    /-
      case mpr
      X : Type u
      h : IsEmpty X
      ⊢ Nonempty (CategoryTheory.Limits.IsInitial X)
    -/
    exact ⟨IsInitial.ofIso Types.isInitialPunit <| Equiv.toIso <| Equiv.equivOfIsEmpty PEmpty X⟩
    /-
      🎉 no goals
    -/


/-- The product type `X × Y` forms a cone for the binary product of `X` and `Y`. -/
@[simps! pt]
def binaryProductCone (X Y : Type u) : BinaryFan X Y :=
  BinaryFan.mk _root_.Prod.fst _root_.Prod.snd


@[simp]
theorem binaryProductCone_fst (X Y : Type u) : (binaryProductCone X Y).fst = _root_.Prod.fst :=
  rfl


@[simp]
theorem binaryProductCone_snd (X Y : Type u) : (binaryProductCone X Y).snd = _root_.Prod.snd :=
  rfl


/-- The product type `X × Y` is a binary product for `X` and `Y`. -/
@[simps]
def binaryProductLimit (X Y : Type u) : IsLimit (binaryProductCone X Y) where
  lift (s : BinaryFan X Y) x := (s.fst x, s.snd x)
  fac _ j := Discrete.recOn j fun j => WalkingPair.casesOn j rfl rfl
  uniq _ _ w := funext fun x => Prod.ext (congr_fun (w ⟨left⟩) x) (congr_fun (w ⟨right⟩) x)


/-- The category of types has `X × Y`, the usual cartesian product,
as the binary product of `X` and `Y`.
-/
@[simps]
def binaryProductLimitCone (X Y : Type u) : Limits.LimitCone (pair X Y) :=
  ⟨_, binaryProductLimit X Y⟩


/-- The categorical binary product in `Type u` is cartesian product. -/
noncomputable def binaryProductIso (X Y : Type u) : Limits.prod X Y ≅ X × Y :=
  limit.isoLimitCone (binaryProductLimitCone X Y)


@[elementwise (attr := simp)]
theorem binaryProductIso_hom_comp_fst (X Y : Type u) :
    (binaryProductIso X Y).hom ≫ _root_.Prod.fst = Limits.prod.fst :=
  limit.isoLimitCone_hom_π (binaryProductLimitCone X Y) ⟨WalkingPair.left⟩


@[elementwise (attr := simp)]
theorem binaryProductIso_hom_comp_snd (X Y : Type u) :
    (binaryProductIso X Y).hom ≫ _root_.Prod.snd = Limits.prod.snd :=
  limit.isoLimitCone_hom_π (binaryProductLimitCone X Y) ⟨WalkingPair.right⟩


@[elementwise (attr := simp)]
theorem binaryProductIso_inv_comp_fst (X Y : Type u) :
    (binaryProductIso X Y).inv ≫ Limits.prod.fst = _root_.Prod.fst :=
  limit.isoLimitCone_inv_π (binaryProductLimitCone X Y) ⟨WalkingPair.left⟩


@[elementwise (attr := simp)]
theorem binaryProductIso_inv_comp_snd (X Y : Type u) :
    (binaryProductIso X Y).inv ≫ Limits.prod.snd = _root_.Prod.snd :=
  limit.isoLimitCone_inv_π (binaryProductLimitCone X Y) ⟨WalkingPair.right⟩

-- Porting note: it was originally @[simps (config := { typeMd := reducible })]
-- We add the option `type_md` to tell `@[simps]` to not treat homomorphisms `X ⟶ Y` in `Type*` as
-- a function type

/-- The functor which sends `X, Y` to the product type `X × Y`. -/
@[simps]
def binaryProductFunctor : Type u ⥤ Type u ⥤ Type u where
  obj X :=
    { obj := fun Y => X × Y
      map := fun { _ Y₂} f => (binaryProductLimit X Y₂).lift
        (BinaryFan.mk _root_.Prod.fst (_root_.Prod.snd ≫ f)) }
  map {X₁ X₂} f :=
    { app := fun Y =>
      (binaryProductLimit X₂ Y).lift (BinaryFan.mk (_root_.Prod.fst ≫ f) _root_.Prod.snd) }


/-- The product functor given by the instance `HasBinaryProducts (Type u)` is isomorphic to the
explicit binary product functor given by the product type.
-/
noncomputable def binaryProductIsoProd : binaryProductFunctor ≅ (prod.functor : Type u ⥤ _) := by
  /-
    ⊢ CategoryTheory.Iso CategoryTheory.Limits.Types.binaryProductFunctor Category …
  -/
  refine NatIso.ofComponents (fun X => ?_) (fun _ => ?_)
    /-
      case refine_1
      X : Type u
      ⊢ CategoryTheory.Iso (CategoryTheory.Limits.Types.binaryProductFunctor.obj X)  …
    -/
  · refine NatIso.ofComponents (fun Y => ?_) (fun _ => ?_)
      /-
        case refine_1.refine_1
        X Y : Type u
        ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Types.binaryProductFunctor.obj X) …
      -/
    · exact ((limit.isLimit _).conePointUniqueUpToIso (binaryProductLimit X Y)).symm
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        X X✝ Y✝ : Type u
        x✝ : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.Types.binaryP …
      -/
                                             /-
                                               🎉 no goals
                                             -/
    · apply Limits.prod.hom_ext <;> simp <;> rfl
                                             /-
                                               🎉 no goals
                                             -/
    /-
      case refine_2
      X✝ Y✝ : Type u
      x✝ : Quiver.Hom X✝ Y✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.binaryPr …
    -/
  · ext : 2
    /-
      case refine_2.w.h
      X✝ Y✝ : Type u
      x✝¹ : Quiver.Hom X✝ Y✝
      x✝ : Type u
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.binaryP …
    -/
                                           /-
                                             🎉 no goals
                                           -/
    apply Limits.prod.hom_ext <;> simp <;> rfl
                                           /-
                                             🎉 no goals
                                           -/


/-- The sum type `X ⊕ Y` forms a cocone for the binary coproduct of `X` and `Y`. -/
@[simps!]
def binaryCoproductCocone (X Y : Type u) : Cocone (pair X Y) :=
  BinaryCofan.mk Sum.inl Sum.inr


/-- The sum type `X ⊕ Y` is a binary coproduct for `X` and `Y`. -/
@[simps]
def binaryCoproductColimit (X Y : Type u) : IsColimit (binaryCoproductCocone X Y) where
  desc := fun s : BinaryCofan X Y => Sum.elim s.inl s.inr
  fac _ j := Discrete.recOn j fun j => WalkingPair.casesOn j rfl rfl
  uniq _ _ w := funext fun x => Sum.casesOn x (congr_fun (w ⟨left⟩)) (congr_fun (w ⟨right⟩))


/-- The category of types has `X ⊕ Y`,
as the binary coproduct of `X` and `Y`.
-/
def binaryCoproductColimitCocone (X Y : Type u) : Limits.ColimitCocone (pair X Y) :=
  ⟨_, binaryCoproductColimit X Y⟩


/-- The categorical binary coproduct in `Type u` is the sum `X ⊕ Y`. -/
noncomputable def binaryCoproductIso (X Y : Type u) : Limits.coprod X Y ≅ X ⊕ Y :=
  colimit.isoColimitCocone (binaryCoproductColimitCocone X Y)

--open CategoryTheory.Type


@[elementwise (attr := simp)]
theorem binaryCoproductIso_inl_comp_hom (X Y : Type u) :
    Limits.coprod.inl ≫ (binaryCoproductIso X Y).hom = Sum.inl :=
  colimit.isoColimitCocone_ι_hom (binaryCoproductColimitCocone X Y) ⟨WalkingPair.left⟩


@[elementwise (attr := simp)]
theorem binaryCoproductIso_inr_comp_hom (X Y : Type u) :
    Limits.coprod.inr ≫ (binaryCoproductIso X Y).hom = Sum.inr :=
  colimit.isoColimitCocone_ι_hom (binaryCoproductColimitCocone X Y) ⟨WalkingPair.right⟩


@[elementwise (attr := simp)]
theorem binaryCoproductIso_inl_comp_inv (X Y : Type u) :
    ↾(Sum.inl : X ⟶ X ⊕ Y) ≫ (binaryCoproductIso X Y).inv = Limits.coprod.inl :=
  colimit.isoColimitCocone_ι_inv (binaryCoproductColimitCocone X Y) ⟨WalkingPair.left⟩


@[elementwise (attr := simp)]
theorem binaryCoproductIso_inr_comp_inv (X Y : Type u) :
    ↾(Sum.inr : Y ⟶ X ⊕ Y) ≫ (binaryCoproductIso X Y).inv = Limits.coprod.inr :=
  colimit.isoColimitCocone_ι_inv (binaryCoproductColimitCocone X Y) ⟨WalkingPair.right⟩


theorem binaryCofan_isColimit_iff {X Y : Type u} (c : BinaryCofan X Y) :
    Nonempty (IsColimit c) ↔
      Injective c.inl ∧ Injective c.inr ∧ IsCompl (Set.range c.inl) (Set.range c.inr) := by
  classical
    constructor
    · rintro ⟨h⟩
      rw [← show _ = c.inl from
          h.comp_coconePointUniqueUpToIso_inv (binaryCoproductColimit X Y) ⟨WalkingPair.left⟩,
        ← show _ = c.inr from
          h.comp_coconePointUniqueUpToIso_inv (binaryCoproductColimit X Y) ⟨WalkingPair.right⟩]
      dsimp [binaryCoproductCocone]
      refine
        ⟨(h.coconePointUniqueUpToIso (binaryCoproductColimit X Y)).symm.toEquiv.injective.comp
            Sum.inl_injective,
          (h.coconePointUniqueUpToIso (binaryCoproductColimit X Y)).symm.toEquiv.injective.comp
            Sum.inr_injective, ?_⟩
      erw [Set.range_comp, ← eq_compl_iff_isCompl, Set.range_comp _ Sum.inr, ←
        Set.image_compl_eq
          (h.coconePointUniqueUpToIso (binaryCoproductColimit X Y)).symm.toEquiv.bijective]
      simp
    · rintro ⟨h₁, h₂, h₃⟩
      have : ∀ x, x ∈ Set.range c.inl ∨ x ∈ Set.range c.inr := by
        rw [eq_compl_iff_isCompl.mpr h₃.symm]
        exact fun _ => or_not
      refine ⟨BinaryCofan.IsColimit.mk _ ?_ ?_ ?_ ?_⟩
      · intro T f g x
        exact
          if h : x ∈ Set.range c.inl then f ((Equiv.ofInjective _ h₁).symm ⟨x, h⟩)
          else g ((Equiv.ofInjective _ h₂).symm ⟨x, (this x).resolve_left h⟩)
      · intro T f g
        funext x
        dsimp
        simp [h₁.eq_iff]
      · intro T f g
        funext x
        dsimp
        simp only [Set.mem_range, Equiv.ofInjective_symm_apply,
          dite_eq_right_iff, forall_exists_index]
        intro y e
        have : c.inr x ∈ Set.range c.inl ⊓ Set.range c.inr := ⟨⟨_, e⟩, ⟨_, rfl⟩⟩
        rw [disjoint_iff.mp h₃.1] at this
        exact this.elim
      · rintro T _ _ m rfl rfl
        funext x
        dsimp
        split_ifs <;> exact congr_arg _ (Equiv.apply_ofInjective_symm _ ⟨_, _⟩).symm


/-- Any monomorphism in `Type` is a coproduct injection. -/
noncomputable def isCoprodOfMono {X Y : Type u} (f : X ⟶ Y) [Mono f] :
    IsColimit (BinaryCofan.mk f (Subtype.val : ↑(Set.range f)ᶜ → Y)) := by
  /-
    X Y : Type u
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan.mk f Subt …
  -/
  apply Nonempty.some
  /-
    case h
    X Y : Type u
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    ⊢ Nonempty (CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.BinaryCofan …
  -/
  rw [binaryCofan_isColimit_iff]
  /-
    case h
    X Y : Type u
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    ⊢ And (Function.Injective (CategoryTheory.Limits.BinaryCofan.mk f Subtype.val) …
  -/
  refine ⟨(mono_iff_injective f).mp inferInstance, Subtype.val_injective, ?_⟩
  /-
    case h
    X Y : Type u
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    ⊢ IsCompl (Set.range (CategoryTheory.Limits.BinaryCofan.mk f Subtype.val).inl) …
  -/
  symm
  /-
    case h
    X Y : Type u
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    ⊢ IsCompl (Set.range (CategoryTheory.Limits.BinaryCofan.mk f Subtype.val).inr) …
  -/
  rw [← eq_compl_iff_isCompl]
  /-
    case h
    X Y : Type u
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Mono f
    ⊢ Eq (Set.range (CategoryTheory.Limits.BinaryCofan.mk f Subtype.val).inr) (Has …
  -/
  exact Subtype.range_val
  /-
    🎉 no goals
  -/


/--
The category of types has `Π j, f j` as the product of a type family `f : J → TypeMax.{v, u}`.
-/
def productLimitCone {J : Type v} (F : J → TypeMax.{v, u}) :
    Limits.LimitCone (Discrete.functor F) where
  cone :=
    { pt := ∀ j, F j
      π := Discrete.natTrans (fun ⟨j⟩ f => f j) }
  isLimit :=
    { lift := fun s x j => s.π.app ⟨j⟩ x
      uniq := fun _ _ w => funext fun x => funext fun j => (congr_fun (w ⟨j⟩) x : _) }


/-- The categorical product in `TypeMax.{v, u}` is the type theoretic product `Π j, F j`. -/
noncomputable def productIso {J : Type v} (F : J → TypeMax.{v, u}) : ∏ᶜ F ≅ ∀ j, F j :=
  limit.isoLimitCone (productLimitCone.{v, u} F)

-- Porting note: was `@[elementwise (attr := simp)]`, but it produces a trivial lemma
-- It should produce the lemma below.

@[simp]
theorem productIso_hom_comp_eval {J : Type v} (F : J → TypeMax.{v, u}) (j : J) :
    ((productIso.{v, u} F).hom ≫ fun f => f j) = Pi.π F j :=
  rfl


@[simp]
theorem productIso_hom_comp_eval_apply {J : Type v} (F : J → TypeMax.{v, u}) (j : J) (x) :
    ((productIso.{v, u} F).hom x) j = Pi.π F j x :=
  rfl


@[elementwise (attr := simp)]
theorem productIso_inv_comp_π {J : Type v} (F : J → TypeMax.{v, u}) (j : J) :
    (productIso.{v, u} F).inv ≫ Pi.π F j = fun f => f j :=
  limit.isoLimitCone_inv_π (productLimitCone.{v, u} F) ⟨j⟩


/--
A variant of `productLimitCone` using a `Small` hypothesis rather than a function to `TypeMax`.
-/
noncomputable def productLimitCone :
    Limits.LimitCone (Discrete.functor F) where
  cone :=
    { pt := Shrink (∀ j, F j)
      π := Discrete.natTrans (fun ⟨j⟩ f => (equivShrink (∀ j, F j)).symm f j) }
  isLimit :=
    have : Small.{u} (∀ j, F j) := inferInstance
    { lift := fun s x => (equivShrink _) (fun j => s.π.app ⟨j⟩ x)
      uniq := fun s m w => funext fun x => Shrink.ext <| funext fun j => by
        /-
          J : Type v
          F : J → Type u
          inst✝ : Small.{u, v} J
          this : Small.{u, max u v} ((j : J) → F j)
          s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor F)
          m : Quiver.Hom s.pt { pt := Shrink.{u, max u v} ((j : J) → F j), π := Category …
          w : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
          x : s.pt
          j : J
          ⊢ Eq ((equivShrink ((j : J) → F j)).symm (m x) j) ((equivShrink ((j : J) → F j …
        -/
        simpa using (congr_fun (w ⟨j⟩) x : _) }
        /-
          🎉 no goals
        -/


/-- The categorical product in `Type u` indexed in `Type v`
is the type theoretic product `Π j, F j`, after shrinking back to `Type u`. -/
noncomputable def productIso :
    (∏ᶜ F : Type u) ≅ Shrink.{u} (∀ j, F j) :=
  limit.isoLimitCone (productLimitCone.{v, u} F)


@[simp]
theorem productIso_hom_comp_eval (j : J) :
    ((productIso.{v, u} F).hom ≫ fun f => (equivShrink (∀ j, F j)).symm f j) = Pi.π F j :=
  limit.isoLimitCone_hom_π (productLimitCone.{v, u} F) ⟨j⟩

-- Porting note:
-- `elementwise` seems to be broken. Applied to the previous lemma, it should produce:

@[simp]
theorem productIso_hom_comp_eval_apply (j : J) (x) :
    (equivShrink (∀ j, F j)).symm ((productIso F).hom x) j = Pi.π F j x :=
  congr_fun (productIso_hom_comp_eval F j) x


@[elementwise (attr := simp)]
theorem productIso_inv_comp_π (j : J) :
    (productIso.{v, u} F).inv ≫ Pi.π F j = fun f => ((equivShrink (∀ j, F j)).symm f) j :=
  limit.isoLimitCone_inv_π (productLimitCone.{v, u} F) ⟨j⟩


/-- The category of types has `Σ j, f j` as the coproduct of a type family `f : J → Type`.
-/
def coproductColimitCocone {J : Type v} (F : J → TypeMax.{v, u}) :
    Limits.ColimitCocone (Discrete.functor F) where
  cocone :=
    { pt := Σj, F j
      ι := Discrete.natTrans (fun ⟨j⟩ x => ⟨j, x⟩)}
  isColimit :=
    { desc := fun s x => s.ι.app ⟨x.1⟩ x.2
      uniq := fun s m w => by
        /-
          J : Type v
          F : J → TypeMax
          s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor F)
          m : Quiver.Hom { pt := Sigma fun j => F j, ι := CategoryTheory.Discrete.natTra …
          w : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
          ⊢ Eq m ((fun s x => s.ι.app { as := x.fst } x.snd) s)
        -/
        funext ⟨j, x⟩
        /-
          case h
          J : Type v
          F : J → TypeMax
          s : CategoryTheory.Limits.Cocone (CategoryTheory.Discrete.functor F)
          m : Quiver.Hom { pt := Sigma fun j => F j, ι := CategoryTheory.Discrete.natTra …
          w : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
          j : J
          x : F j
          ⊢ Eq (m ⟨j, x⟩) ((fun s x => s.ι.app { as := x.fst } x.snd) s ⟨j, x⟩)
        -/
        exact congr_fun (w ⟨j⟩) x }
        /-
          🎉 no goals
        -/


/-- The categorical coproduct in `Type u` is the type theoretic coproduct `Σ j, F j`. -/
noncomputable def coproductIso {J : Type v} (F : J → TypeMax.{v, u}) : ∐ F ≅ Σj, F j :=
  colimit.isoColimitCocone (coproductColimitCocone F)


@[elementwise (attr := simp)]
theorem coproductIso_ι_comp_hom {J : Type v} (F : J → TypeMax.{v, u}) (j : J) :
    Sigma.ι F j ≫ (coproductIso F).hom = fun x : F j => (⟨j, x⟩ : Σj, F j) :=
  colimit.isoColimitCocone_ι_hom (coproductColimitCocone F) ⟨j⟩

-- Porting note: was @[elementwise (attr := simp)], but it produces a trivial lemma
-- removed simp attribute because it seems it never applies

theorem coproductIso_mk_comp_inv {J : Type v} (F : J → TypeMax.{v, u}) (j : J) :
    (↾fun x : F j => (⟨j, x⟩ : Σj, F j)) ≫ (coproductIso F).inv = Sigma.ι F j :=
  rfl


/--
Show the given fork in `Type u` is an equalizer given that any element in the "difference kernel"
comes from `X`.
The converse of `unique_of_type_equalizer`.
-/
noncomputable def typeEqualizerOfUnique (t : ∀ y : Y, g y = h y → ∃! x : X, f x = y) :
    IsLimit (Fork.ofι _ w) :=
  Fork.IsLimit.mk' _ fun s => by
    /-
      X Y Z : Type u
      f : Quiver.Hom X Y
      g h : Quiver.Hom Y Z
      w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
      t : ∀ (y : Y), Eq (g y) (h y) → ExistsUnique fun x => Eq (f x) y
      s : CategoryTheory.Limits.Fork g h
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheo …
    -/
    refine ⟨fun i => ?_, ?_, ?_⟩
      /-
        case refine_1
        X Y Z : Type u
        f : Quiver.Hom X Y
        g h : Quiver.Hom Y Z
        w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
        t : ∀ (y : Y), Eq (g y) (h y) → ExistsUnique fun x => Eq (f x) y
        s : CategoryTheory.Limits.Fork g h
        i : ((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingParallelPair). …
        ⊢ ((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingParallelPair).ob …
      -/
    · apply Classical.choose (t (s.ι i) _)
      /-
        X Y Z : Type u
        f : Quiver.Hom X Y
        g h : Quiver.Hom Y Z
        w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
        t : ∀ (y : Y), Eq (g y) (h y) → ExistsUnique fun x => Eq (f x) y
        s : CategoryTheory.Limits.Fork g h
        i : ((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingParallelPair). …
        ⊢ Eq (g (s.ι i)) (h (s.ι i))
      -/
      apply congr_fun s.condition i
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        X Y Z : Type u
        f : Quiver.Hom X Y
        g h : Quiver.Hom Y Z
        w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
        t : ∀ (y : Y), Eq (g y) (h y) → ExistsUnique fun x => Eq (f x) y
        s : CategoryTheory.Limits.Fork g h
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun i => Classical.choose ⋯) (Catego …
      -/
    · funext i
      /-
        case refine_2.h
        X Y Z : Type u
        f : Quiver.Hom X Y
        g h : Quiver.Hom Y Z
        w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
        t : ∀ (y : Y), Eq (g y) (h y) → ExistsUnique fun x => Eq (f x) y
        s : CategoryTheory.Limits.Fork g h
        i : ((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingParallelPair). …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun i => Classical.choose ⋯) (Catego …
      -/
      exact (Classical.choose_spec (t (s.ι i) (congr_fun s.condition i))).1
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        X Y Z : Type u
        f : Quiver.Hom X Y
        g h : Quiver.Hom Y Z
        w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
        t : ∀ (y : Y), Eq (g y) (h y) → ExistsUnique fun x => Eq (f x) y
        s : CategoryTheory.Limits.Fork g h
        ⊢ ∀ {m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.Walk …
      -/
    · intro m hm
      /-
        case refine_3
        X Y Z : Type u
        f : Quiver.Hom X Y
        g h : Quiver.Hom Y Z
        w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
        t : ∀ (y : Y), Eq (g y) (h y) → ExistsUnique fun x => Eq (f x) y
        s : CategoryTheory.Limits.Fork g h
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ofι  …
        ⊢ Eq m fun i => Classical.choose ⋯
      -/
      funext i
      /-
        case refine_3.h
        X Y Z : Type u
        f : Quiver.Hom X Y
        g h : Quiver.Hom Y Z
        w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
        t : ∀ (y : Y), Eq (g y) (h y) → ExistsUnique fun x => Eq (f x) y
        s : CategoryTheory.Limits.Fork g h
        m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
        hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ofι  …
        i : ((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingParallelPair). …
        ⊢ Eq (m i) (Classical.choose ⋯)
      -/
      exact (Classical.choose_spec (t (s.ι i) (congr_fun s.condition i))).2 _ (congr_fun hm i)
      /-
        🎉 no goals
      -/


/-- The converse of `type_equalizer_of_unique`. -/
theorem unique_of_type_equalizer (t : IsLimit (Fork.ofι _ w)) (y : Y) (hy : g y = h y) :
    ∃! x : X, f x = y := by
  /-
    X Y Z : Type u
    f : Quiver.Hom X Y
    g h : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
    t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι f w)
    y : Y
    hy : Eq (g y) (h y)
    ⊢ ExistsUnique fun x => Eq (f x) y
  -/
  let y' : PUnit ⟶ Y := fun _ => y
  /-
    X Y Z : Type u
    f : Quiver.Hom X Y
    g h : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
    t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι f w)
    y : Y
    hy : Eq (g y) (h y)
    y' : Quiver.Hom PUnit.{u + 1} Y := fun x => y
    ⊢ ExistsUnique fun x => Eq (f x) y
  -/
  have hy' : y' ≫ g = y' ≫ h := funext fun _ => hy
  /-
    X Y Z : Type u
    f : Quiver.Hom X Y
    g h : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
    t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι f w)
    y : Y
    hy : Eq (g y) (h y)
    y' : Quiver.Hom PUnit.{u + 1} Y := fun x => y
    hy' : Eq (CategoryTheory.CategoryStruct.comp y' g) (CategoryTheory.CategoryStr …
    ⊢ ExistsUnique fun x => Eq (f x) y
  -/
  refine ⟨(Fork.IsLimit.lift' t _ hy').1 ⟨⟩, congr_fun (Fork.IsLimit.lift' t y' _).2 ⟨⟩, ?_⟩
  /-
    X Y Z : Type u
    f : Quiver.Hom X Y
    g h : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
    t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι f w)
    y : Y
    hy : Eq (g y) (h y)
    y' : Quiver.Hom PUnit.{u + 1} Y := fun x => y
    hy' : Eq (CategoryTheory.CategoryStruct.comp y' g) (CategoryTheory.CategoryStr …
    ⊢ ∀ (y_1 : X), (fun x => Eq (f x) y) y_1 → Eq y_1 (↑(CategoryTheory.Limits.For …
  -/
  intro x' hx'
  suffices (fun _ : PUnit => x') = (Fork.IsLimit.lift' t y' hy').1 by
    rw [← this]
  /-
    X Y Z : Type u
    f : Quiver.Hom X Y
    g h : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
    t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι f w)
    y : Y
    hy : Eq (g y) (h y)
    y' : Quiver.Hom PUnit.{u + 1} Y := fun x => y
    hy' : Eq (CategoryTheory.CategoryStruct.comp y' g) (CategoryTheory.CategoryStr …
    x' : X
    hx' : Eq (f x') y
    ⊢ Eq (fun x => x') ↑(CategoryTheory.Limits.Fork.IsLimit.lift' t y' hy')
  -/
  apply Fork.IsLimit.hom_ext t
  /-
    X Y Z : Type u
    f : Quiver.Hom X Y
    g h : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
    t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι f w)
    y : Y
    hy : Eq (g y) (h y)
    y' : Quiver.Hom PUnit.{u + 1} Y := fun x => y
    hy' : Eq (CategoryTheory.CategoryStruct.comp y' g) (CategoryTheory.CategoryStr …
    x' : X
    hx' : Eq (f x') y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun x => x') (CategoryTheory.Limits. …
  -/
  funext ⟨⟩
  /-
    case h
    X Y Z : Type u
    f : Quiver.Hom X Y
    g h : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
    t : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.Fork.ofι f w)
    y : Y
    hy : Eq (g y) (h y)
    y' : Quiver.Hom PUnit.{u + 1} Y := fun x => y
    hy' : Eq (CategoryTheory.CategoryStruct.comp y' g) (CategoryTheory.CategoryStr …
    x' : X
    hx' : Eq (f x') y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun x => x') (CategoryTheory.Limits. …
  -/
  apply hx'.trans (congr_fun (Fork.IsLimit.lift' t _ hy').2 ⟨⟩).symm
  /-
    🎉 no goals
  -/


theorem type_equalizer_iff_unique :
    Nonempty (IsLimit (Fork.ofι _ w)) ↔ ∀ y : Y, g y = h y → ∃! x : X, f x = y :=
  ⟨fun i => unique_of_type_equalizer _ _ (Classical.choice i), fun k =>
    ⟨typeEqualizerOfUnique f w k⟩⟩


/-- Show that the subtype `{x : Y // g x = h x}` is an equalizer for the pair `(g,h)`. -/
def equalizerLimit : Limits.LimitCone (parallelPair g h) where
  cone := Fork.ofι (Subtype.val : { x : Y // g x = h x } → Y) (funext Subtype.prop)
  isLimit :=
    Fork.IsLimit.mk' _ fun s =>
                           /-
                             X Y Z : Type u
                             f : Quiver.Hom X Y
                             g h : Quiver.Hom Y Z
                             w : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.CategoryStruct …
                             s : CategoryTheory.Limits.Fork g h
                             i : ((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingParallelPair). …
                             ⊢ Eq (g (s.ι i)) (h (s.ι i))
                           -/
      ⟨fun i => ⟨s.ι i, by apply congr_fun s.condition i⟩, rfl, fun hm =>
                           /-
                             🎉 no goals
                           -/
        funext fun x => Subtype.ext (congr_fun hm x)⟩


/-- The categorical equalizer in `Type u` is `{x : Y // g x = h x}`. -/
noncomputable def equalizerIso : equalizer g h ≅ { x : Y // g x = h x } :=
  limit.isoLimitCone equalizerLimit

-- Porting note: was @[elementwise], but it produces a trivial lemma

@[simp]
theorem equalizerIso_hom_comp_subtype : (equalizerIso g h).hom ≫ Subtype.val = equalizer.ι g h := by
  /-
    Y Z : Type u
    g h : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.equalize …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[elementwise (attr := simp)]
theorem equalizerIso_inv_comp_ι : (equalizerIso g h).inv ≫ equalizer.ι g h = Subtype.val :=
  limit.isoLimitCone_inv_π equalizerLimit WalkingParallelPair.zero


/-- (Implementation) The relation to be quotiented to obtain the coequalizer. -/
inductive CoequalizerRel : Y → Y → Prop
  | Rel (x : X) : CoequalizerRel (f x) (g x)


/-- Show that the quotient by the relation generated by `f(x) ~ g(x)`
is a coequalizer for the pair `(f, g)`.
-/
def coequalizerColimit : Limits.ColimitCocone (parallelPair f g) where
  cocone :=
    Cofork.ofπ (Quot.mk (CoequalizerRel f g)) (funext fun x => Quot.sound (CoequalizerRel.Rel x))
  isColimit :=
    Cofork.IsColimit.mk _
      (fun s => Quot.lift s.π
        (fun a b (h : CoequalizerRel f g a b) => by
          /-
            X Y Z : Type u
            f g : Quiver.Hom X Y
            s : CategoryTheory.Limits.Cofork f g
            a b : Y
            h : CategoryTheory.Limits.Types.CoequalizerRel f g a b
            ⊢ Eq (s.π a) (s.π b)
          -/
          cases h
          /-
            case Rel
            X Y Z : Type u
            f g : Quiver.Hom X Y
            s : CategoryTheory.Limits.Cofork f g
            x✝ : X
            ⊢ Eq (s.π (f x✝)) (s.π (g x✝))
          -/
          apply congr_fun s.condition))
          /-
            🎉 no goals
          -/
      (fun _ => rfl)
      (fun _ _ hm => funext (fun x => Quot.inductionOn x (congr_fun hm)))


/-- If `π : Y ⟶ Z` is an equalizer for `(f, g)`, and `U ⊆ Y` such that `f ⁻¹' U = g ⁻¹' U`,
then `π ⁻¹' (π '' U) = U`.
-/
theorem coequalizer_preimage_image_eq_of_preimage_eq (π : Y ⟶ Z) (e : f ≫ π = g ≫ π)
    (h : IsColimit (Cofork.ofπ π e)) (U : Set Y) (H : f ⁻¹' U = g ⁻¹' U) : π ⁻¹' (π '' U) = U := by
  have lem : ∀ x y, CoequalizerRel f g x y → (x ∈ U ↔ y ∈ U) := by
    rintro _ _ ⟨x⟩
    change x ∈ f ⁻¹' U ↔ x ∈ g ⁻¹' U
    rw [H]
  -- Porting note: tidy was able to fill the structure automatically
  have eqv : _root_.Equivalence fun x y => x ∈ U ↔ y ∈ U :=
    { refl := by tauto
      symm := by tauto
      trans := by tauto }
  /-
    X Y Z : Type u
    f g : Quiver.Hom X Y
    π : Quiver.Hom Y Z
    e : Eq (CategoryTheory.CategoryStruct.comp f π) (CategoryTheory.CategoryStruct …
    h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ π e)
    U : Set Y
    H : Eq (Set.preimage f U) (Set.preimage g U)
    lem : ∀ (x y : Y), CategoryTheory.Limits.Types.CoequalizerRel f g x y → Iff (M …
    eqv : _root_.Equivalence fun x y => Iff (Membership.mem U x) (Membership.mem U …
    ⊢ Eq (Set.preimage π (Set.image π U)) U
  -/
  ext
  /-
    case h
    X Y Z : Type u
    f g : Quiver.Hom X Y
    π : Quiver.Hom Y Z
    e : Eq (CategoryTheory.CategoryStruct.comp f π) (CategoryTheory.CategoryStruct …
    h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ π e)
    U : Set Y
    H : Eq (Set.preimage f U) (Set.preimage g U)
    lem : ∀ (x y : Y), CategoryTheory.Limits.Types.CoequalizerRel f g x y → Iff (M …
    eqv : _root_.Equivalence fun x y => Iff (Membership.mem U x) (Membership.mem U …
    x✝ : Y
    ⊢ Iff (Membership.mem (Set.preimage π (Set.image π U)) x✝) (Membership.mem U x✝)
  -/
  constructor
  · rw [←
      show _ = π from
        h.comp_coconePointUniqueUpToIso_inv (coequalizerColimit f g).2
          WalkingParallelPair.one]
    /-
      case h.mp
      X Y Z : Type u
      f g : Quiver.Hom X Y
      π : Quiver.Hom Y Z
      e : Eq (CategoryTheory.CategoryStruct.comp f π) (CategoryTheory.CategoryStruct …
      h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ π e)
      U : Set Y
      H : Eq (Set.preimage f U) (Set.preimage g U)
      lem : ∀ (x y : Y), CategoryTheory.Limits.Types.CoequalizerRel f g x y → Iff (M …
      eqv : _root_.Equivalence fun x y => Iff (Membership.mem U x) (Membership.mem U …
      x✝ : Y
      ⊢ Membership.mem (Set.preimage (CategoryTheory.CategoryStruct.comp ((CategoryT …
    -/
    rintro ⟨y, hy, e'⟩
    /-
      case h.mp.intro.intro
      X Y Z : Type u
      f g : Quiver.Hom X Y
      π : Quiver.Hom Y Z
      e : Eq (CategoryTheory.CategoryStruct.comp f π) (CategoryTheory.CategoryStruct …
      h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ π e)
      U : Set Y
      H : Eq (Set.preimage f U) (Set.preimage g U)
      lem : ∀ (x y : Y), CategoryTheory.Limits.Types.CoequalizerRel f g x y → Iff (M …
      eqv : _root_.Equivalence fun x y => Iff (Membership.mem U x) (Membership.mem U …
      x✝ y : Y
      hy : Membership.mem U y
      e' : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.Types.coeq …
      ⊢ Membership.mem U x✝
    -/
    dsimp at e'
    replace e' :=
      (mono_iff_injective
            (h.coconePointUniqueUpToIso (coequalizerColimit f g).isColimit).inv).mp
        inferInstance e'
    /-
      case h.mp.intro.intro
      X Y Z : Type u
      f g : Quiver.Hom X Y
      π : Quiver.Hom Y Z
      e : Eq (CategoryTheory.CategoryStruct.comp f π) (CategoryTheory.CategoryStruct …
      h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ π e)
      U : Set Y
      H : Eq (Set.preimage f U) (Set.preimage g U)
      lem : ∀ (x y : Y), CategoryTheory.Limits.Types.CoequalizerRel f g x y → Iff (M …
      eqv : _root_.Equivalence fun x y => Iff (Membership.mem U x) (Membership.mem U …
      x✝ y : Y
      hy : Membership.mem U y
      e' : Eq (CategoryTheory.Limits.Cofork.π (CategoryTheory.Limits.Types.coequaliz …
      ⊢ Membership.mem U x✝
    -/
    exact (eqv.eqvGen_iff.mp (Relation.EqvGen.mono lem (Quot.eqvGen_exact e'))).mp hy
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      X Y Z : Type u
      f g : Quiver.Hom X Y
      π : Quiver.Hom Y Z
      e : Eq (CategoryTheory.CategoryStruct.comp f π) (CategoryTheory.CategoryStruct …
      h : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.Cofork.ofπ π e)
      U : Set Y
      H : Eq (Set.preimage f U) (Set.preimage g U)
      lem : ∀ (x y : Y), CategoryTheory.Limits.Types.CoequalizerRel f g x y → Iff (M …
      eqv : _root_.Equivalence fun x y => Iff (Membership.mem U x) (Membership.mem U …
      x✝ : Y
      ⊢ Membership.mem U x✝ → Membership.mem (Set.preimage π (Set.image π U)) x✝
    -/
  · exact fun hx => ⟨_, hx, rfl⟩
    /-
      🎉 no goals
    -/


/-- The categorical coequalizer in `Type u` is the quotient by `f g ~ g x`. -/
noncomputable def coequalizerIso : coequalizer f g ≅ _root_.Quot (CoequalizerRel f g) :=
  colimit.isoColimitCocone (coequalizerColimit f g)


@[elementwise (attr := simp)]
theorem coequalizerIso_π_comp_hom :
    coequalizer.π f g ≫ (coequalizerIso f g).hom = Quot.mk (CoequalizerRel f g) :=
  colimit.isoColimitCocone_ι_hom (coequalizerColimit f g) WalkingParallelPair.one

-- Porting note: was @[elementwise], but it produces a trivial lemma

@[simp]
theorem coequalizerIso_quot_comp_inv :
    ↾Quot.mk (CoequalizerRel f g) ≫ (coequalizerIso f g).inv = coequalizer.π f g :=
  rfl


instance : HasPullbacks.{u} (Type u) :=
  -- FIXME does not work via `inferInstance` despite `#synth HasPullbacks.{u} (Type u)` succeeding.
  -- https://github.com/leanprover-community/mathlib4/issues/5752
  -- inferInstance
  hasPullbacks_of_hasWidePullbacks.{u} (Type u)


instance : HasPushouts.{u} (Type u) :=
  hasPushouts_of_hasWidePushouts.{u} (Type u)


/-- The usual explicit pullback in the category of types, as a subtype of the product.
The full `LimitCone` data is bundled as `pullbackLimitCone f g`.
-/
abbrev PullbackObj : Type u :=
  { p : X × Y // f p.1 = g p.2 }

-- `PullbackObj f g` comes with a coercion to the product type `X × Y`.

/-- The explicit pullback cone on `PullbackObj f g`.
This is bundled with the `IsLimit` data as `pullbackLimitCone f g`.
-/
abbrev pullbackCone : Limits.PullbackCone f g :=
  PullbackCone.mk (fun p : PullbackObj f g => p.1.1) (fun p => p.1.2) (funext fun p => p.2)


/-- The explicit pullback in the category of types, bundled up as a `LimitCone`
for given `f` and `g`.
-/
@[simps]
def pullbackLimitCone (f : X ⟶ Z) (g : Y ⟶ Z) : Limits.LimitCone (cospan f g) where
  cone := pullbackCone f g
  isLimit :=
    PullbackCone.isLimitAux _ (fun s x => ⟨⟨s.fst x, s.snd x⟩, congr_fun s.condition x⟩)
          /-
            X Y Z : Type u
            X' Y' Z' : Type v
            f✝ : Quiver.Hom X Z
            g✝ : Quiver.Hom Y Z
            f' : Quiver.Hom X' Z'
            g' : Quiver.Hom Y' Z'
            f : Quiver.Hom X Z
            g : Quiver.Hom Y Z
            ⊢ ∀ (s : CategoryTheory.Limits.PullbackCone f g), Eq (CategoryTheory.CategoryS …
          -/
          /-
            🎉 no goals
          -/
      (by aesop) (by aesop) fun _ _ w =>
                     /-
                       🎉 no goals
                     -/
      funext fun x =>
        Subtype.ext <|
          Prod.ext (congr_fun (w WalkingCospan.left) x) (congr_fun (w WalkingCospan.right) x)


/-- A limit pullback cone in the category of types identifies to the explicit pullback. -/
noncomputable def equivPullbackObj : c.pt ≃ Types.PullbackObj f g :=
  (IsLimit.conePointUniqueUpToIso hc (Types.pullbackLimitCone f g).isLimit).toEquiv


@[simp]
lemma equivPullbackObj_apply_fst (x : c.pt) : (equivPullbackObj hc x).1.1 = c.fst x :=
  congr_fun (IsLimit.conePointUniqueUpToIso_hom_comp hc
    (Types.pullbackLimitCone f g).isLimit .left) x


@[simp]
lemma equivPullbackObj_apply_snd (x : c.pt) : (equivPullbackObj hc x).1.2 = c.snd x :=
  congr_fun (IsLimit.conePointUniqueUpToIso_hom_comp hc
    (Types.pullbackLimitCone f g).isLimit .right) x


@[simp]
lemma equivPullbackObj_symm_apply_fst (x : Types.PullbackObj f g) :
    c.fst ((equivPullbackObj hc).symm x) = x.1.1 := by
  /-
    X Y S : Type v
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    c : CategoryTheory.Limits.PullbackCone f g
    hc : CategoryTheory.Limits.IsLimit c
    x : CategoryTheory.Limits.Types.PullbackObj f g
    ⊢ Eq (c.fst ((CategoryTheory.Limits.PullbackCone.IsLimit.equivPullbackObj hc). …
  -/
  obtain ⟨x, rfl⟩ := (equivPullbackObj hc).surjective x
  /-
    case intro
    X Y S : Type v
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    c : CategoryTheory.Limits.PullbackCone f g
    hc : CategoryTheory.Limits.IsLimit c
    x : c.pt
    ⊢ Eq (c.fst ((CategoryTheory.Limits.PullbackCone.IsLimit.equivPullbackObj hc). …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma equivPullbackObj_symm_apply_snd (x : Types.PullbackObj f g) :
    c.snd ((equivPullbackObj hc).symm x) = x.1.2 := by
  /-
    X Y S : Type v
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    c : CategoryTheory.Limits.PullbackCone f g
    hc : CategoryTheory.Limits.IsLimit c
    x : CategoryTheory.Limits.Types.PullbackObj f g
    ⊢ Eq (c.snd ((CategoryTheory.Limits.PullbackCone.IsLimit.equivPullbackObj hc). …
  -/
  obtain ⟨x, rfl⟩ := (equivPullbackObj hc).surjective x
  /-
    case intro
    X Y S : Type v
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    c : CategoryTheory.Limits.PullbackCone f g
    hc : CategoryTheory.Limits.IsLimit c
    x : c.pt
    ⊢ Eq (c.snd ((CategoryTheory.Limits.PullbackCone.IsLimit.equivPullbackObj hc). …
  -/
  simp
  /-
    🎉 no goals
  -/


include hc in
lemma type_ext {x y : c.pt} (h₁ : c.fst x = c.fst y) (h₂ : c.snd x = c.snd y) : x = y :=
                                      /-
                                        X Y S : Type v
                                        f : Quiver.Hom X S
                                        g : Quiver.Hom Y S
                                        c : CategoryTheory.Limits.PullbackCone f g
                                        hc : CategoryTheory.Limits.IsLimit c
                                        x y : c.pt
                                        h₁ : Eq (c.fst x) (c.fst y)
                                        h₂ : Eq (c.snd x) (c.snd y)
                                        ⊢ Eq ((CategoryTheory.Limits.PullbackCone.IsLimit.equivPullbackObj hc) x) ((Ca …
                                      -/
                                              /-
                                                🎉 no goals
                                              -/
  (equivPullbackObj hc).injective (by ext <;> assumption)
                                              /-
                                                🎉 no goals
                                              -/


/-- Given `c : PullbackCone f g` in the category of types, this is
the canonical map `c.pt → Types.PullbackObj f g`. -/
@[simps coe_fst coe_snd]
def toPullbackObj (x : c.pt) : Types.PullbackObj f g :=
  ⟨⟨c.fst x, c.snd x⟩, congr_fun c.condition x⟩


/-- A pullback cone `c` in the category of types is limit iff the
map `c.toPullbackObj : c.pt → Types.PullbackObj f g` is a bijection. -/
noncomputable def isLimitEquivBijective :
    IsLimit c ≃ Function.Bijective c.toPullbackObj where
  toFun h := (IsLimit.equivPullbackObj h).bijective
  invFun h := IsLimit.ofIsoLimit (Types.pullbackLimitCone f g).isLimit
               /-
                 X Y S : Type v
                 f : Quiver.Hom X S
                 g : Quiver.Hom Y S
                 c : CategoryTheory.Limits.PullbackCone f g
                 h : Function.Bijective c.toPullbackObj
                 ⊢ Eq c.fst (CategoryTheory.CategoryStruct.comp (Equiv.ofBijective c.toPullback …
               -/
               /-
                 🎉 no goals
               -/
    (Iso.symm (PullbackCone.ext (Equiv.ofBijective _ h).toIso))
               /-
                 🎉 no goals
               -/
  left_inv _ := Subsingleton.elim _ _
  right_inv _ := rfl


/-- The pullback given by the instance `HasPullbacks (Type u)` is isomorphic to the
explicit pullback object given by `PullbackObj`.
-/
noncomputable def pullbackIsoPullback : pullback f g ≅ PullbackObj f g :=
  (PullbackCone.IsLimit.equivPullbackObj (pullbackIsPullback f g)).toIso


@[simp]
theorem pullbackIsoPullback_hom_fst (p : pullback f g) :
    ((pullbackIsoPullback f g).hom p : X × Y).fst = (pullback.fst f g) p :=
  PullbackCone.IsLimit.equivPullbackObj_apply_fst (pullbackIsPullback f g) p


@[simp]
theorem pullbackIsoPullback_hom_snd (p : pullback f g) :
    ((pullbackIsoPullback f g).hom p : X × Y).snd = (pullback.snd f g) p :=
  PullbackCone.IsLimit.equivPullbackObj_apply_snd (pullbackIsPullback f g) p


@[simp]
theorem pullbackIsoPullback_inv_fst_apply (x : (Types.pullbackCone f g).pt) :
    (pullback.fst f g) ((pullbackIsoPullback f g).inv x) = (fun p => (p.1 : X × Y).fst) x :=
  PullbackCone.IsLimit.equivPullbackObj_symm_apply_fst (pullbackIsPullback f g) x


@[simp]
theorem pullbackIsoPullback_inv_snd_apply (x : (Types.pullbackCone f g).pt) :
    (pullback.snd f g) ((pullbackIsoPullback f g).inv x) = (fun p => (p.1 : X × Y).snd) x :=
  PullbackCone.IsLimit.equivPullbackObj_symm_apply_snd (pullbackIsPullback f g) x


@[simp]
theorem pullbackIsoPullback_inv_fst :
                                                                                        /-
                                                                                          X Y Z : Type u
                                                                                          f : Quiver.Hom X Z
                                                                                          g : Quiver.Hom Y Z
                                                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.pullback …
                                                                                        -/
    (pullbackIsoPullback f g).inv ≫ pullback.fst _ _ = fun p => (p.1 : X × Y).fst := by aesop
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


@[simp]
theorem pullbackIsoPullback_inv_snd :
                                                                                        /-
                                                                                          X Y Z : Type u
                                                                                          f : Quiver.Hom X Z
                                                                                          g : Quiver.Hom Y Z
                                                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.pullback …
                                                                                        -/
    (pullbackIsoPullback f g).inv ≫ pullback.snd _ _ = fun p => (p.1 : X × Y).snd := by aesop
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


/-- The pushout of two maps `f : S ⟶ X₁` and `g : S ⟶ X₂` is the quotient
by the equivalence relation on `X₁ ⊕ X₂` generated by this relation. -/
inductive Pushout.Rel (f : S ⟶ X₁) (g : S ⟶ X₂) : X₁ ⊕ X₂ → X₁ ⊕ X₂ → Prop
  | inl_inr (s : S) : Pushout.Rel f g (Sum.inl (f s)) (Sum.inr (g s))


/-- Construction of the pushout in the category of types, as a quotient of `X₁ ⊕ X₂`. -/
def Pushout : Type u := _root_.Quot (Pushout.Rel f g)


/-- In case `f : S ⟶ X₁` is a monomorphism, this relation is the equivalence relation
generated by `Pushout.Rel f g`. -/
inductive Pushout.Rel' : X₁ ⊕ X₂ → X₁ ⊕ X₂ → Prop
  | refl (x : X₁ ⊕ X₂) : Rel' x x
  | inl_inl (x₀ y₀ : S) (h : g x₀ = g y₀) : Rel' (Sum.inl (f x₀)) (Sum.inl (f y₀))
  | inl_inr (s : S) : Rel' (Sum.inl (f s)) (Sum.inr (g s))
  | inr_inl (s : S) : Rel' (Sum.inr (g s)) (Sum.inl (f s))


/-- The quotient of `X₁ ⊕ X₂` by the relation `PushoutRel' f g`. -/
def Pushout' : Type u := _root_.Quot (Pushout.Rel' f g)


/-- The left inclusion in the constructed pushout `Pushout f g`. -/
@[simp]
def inl : X₁ ⟶ Pushout f g := fun x => Quot.mk _ (Sum.inl x)


/-- The right inclusion in the constructed pushout `Pushout f g`. -/
@[simp]
def inr : X₂ ⟶ Pushout f g := fun x => Quot.mk _ (Sum.inr x)


lemma condition : f ≫ inl f g = g ≫ inr f g := by
  /-
    S X₁ X₂ : Type u
    f : Quiver.Hom S X₁
    g : Quiver.Hom S X₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Types.Pushou …
  -/
  ext x
  /-
    case h
    S X₁ X₂ : Type u
    f : Quiver.Hom S X₁
    g : Quiver.Hom S X₂
    x : S
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Types.Pushou …
  -/
  exact Quot.sound (Rel.inl_inr x)
  /-
    🎉 no goals
  -/


/-- The constructed pushout cocone in the category of types. -/
@[simps!]
def cocone : PushoutCocone f g := PushoutCocone.mk _ _ (condition f g)


/-- The cocone `cocone f g` is colimit. -/
def isColimitCocone : IsColimit (cocone f g) :=
  PushoutCocone.IsColimit.mk _ (fun s => Quot.lift (fun x => match x with
      | Sum.inl x₁ => s.inl x₁
      | Sum.inr x₂ => s.inr x₂) (by
    /-
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      s : CategoryTheory.Limits.PushoutCocone f g
      ⊢ ∀ (a b : Sum X₁ X₂), CategoryTheory.Limits.Types.Pushout.Rel f g a b → Eq (( …
    -/
    rintro _ _ ⟨t⟩
    /-
      case inl_inr
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      s : CategoryTheory.Limits.PushoutCocone f g
      t : S
      ⊢ Eq ((fun x => CategoryTheory.Limits.Types.Pushout.isColimitCocone.match_1 (f …
    -/
    exact congr_fun s.condition t)) (fun _ => rfl) (fun _ => rfl) (fun s m h₁ h₂ => by
    /-
      🎉 no goals
    -/
      /-
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        s : CategoryTheory.Limits.PushoutCocone f g
        m : Quiver.Hom (CategoryTheory.Limits.Types.Pushout f g) s.pt
        h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.Pusho …
        h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.Pusho …
        ⊢ Eq m ((fun s => Quot.lift (fun x => CategoryTheory.Limits.Types.Pushout.isCo …
      -/
      ext ⟨x₁|x₂⟩
        /-
          case h.mk.inl
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          s : CategoryTheory.Limits.PushoutCocone f g
          m : Quiver.Hom (CategoryTheory.Limits.Types.Pushout f g) s.pt
          h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.Pusho …
          h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.Pusho …
          a✝ : CategoryTheory.Limits.Types.Pushout f g
          x₁ : X₁
          ⊢ Eq (m (Quot.mk (CategoryTheory.Limits.Types.Pushout.Rel f g) (Sum.inl x₁)))  …
        -/
      · exact congr_fun h₁ x₁
        /-
          🎉 no goals
        -/
        /-
          case h.mk.inr
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          s : CategoryTheory.Limits.PushoutCocone f g
          m : Quiver.Hom (CategoryTheory.Limits.Types.Pushout f g) s.pt
          h₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.Pusho …
          h₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Types.Pusho …
          a✝ : CategoryTheory.Limits.Types.Pushout f g
          x₂ : X₂
          ⊢ Eq (m (Quot.mk (CategoryTheory.Limits.Types.Pushout.Rel f g) (Sum.inr x₂)))  …
        -/
      · exact congr_fun h₂ x₂)
        /-
          🎉 no goals
        -/


@[simp]
lemma inl_rel'_inl_iff (x₁ y₁ : X₁) :
    Rel' f g (Sum.inl x₁) (Sum.inl y₁) ↔ x₁ = y₁ ∨
      ∃ (x₀ y₀ : S) (_ : g x₀ = g y₀), x₁ = f x₀ ∧ y₁ = f y₀ := by
  /-
    S X₁ X₂ : Type u
    f : Quiver.Hom S X₁
    g : Quiver.Hom S X₂
    x₁ y₁ : X₁
    ⊢ Iff (CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl x₁) (Sum.inl y₁)) …
  -/
  constructor
    /-
      case mp
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      x₁ y₁ : X₁
      ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl x₁) (Sum.inl y₁) → Or  …
    -/
  · rintro (_|⟨_, _, h⟩)
      /-
        case mp.refl
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        x₁ : X₁
        ⊢ Or (Eq x₁ x₁) (Exists fun x₀ => Exists fun y₀ => Exists fun x => And (Eq x₁  …
      -/
    · exact Or.inl rfl
      /-
        🎉 no goals
      -/
      /-
        case mp.inl_inl
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        x₀✝ y₀✝ : S
        h : Eq (g x₀✝) (g y₀✝)
        ⊢ Or (Eq (f x₀✝) (f y₀✝)) (Exists fun x₀ => Exists fun y₀ => Exists fun x => A …
      -/
    · exact Or.inr ⟨_, _, h, rfl, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      x₁ y₁ : X₁
      ⊢ Or (Eq x₁ y₁) (Exists fun x₀ => Exists fun y₀ => Exists fun x => And (Eq x₁  …
    -/
  · rintro (rfl | ⟨_,_ , h, rfl, rfl⟩)
      /-
        case mpr.inl
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        x₁ : X₁
        ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl x₁) (Sum.inl x₁)
      -/
    · apply Rel'.refl
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr.intro.intro.intro.intro
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        w✝¹ w✝ : S
        h : Eq (g w✝¹) (g w✝)
        ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f w✝¹)) (Sum.inl (f w …
      -/
    · exact Rel'.inl_inl _ _ h
      /-
        🎉 no goals
      -/


@[simp]
lemma inl_rel'_inr_iff (x₁ : X₁) (x₂ : X₂) :
    Rel' f g (Sum.inl x₁) (Sum.inr x₂) ↔
      ∃ (s : S), x₁ = f s ∧ x₂ = g s := by
  /-
    S X₁ X₂ : Type u
    f : Quiver.Hom S X₁
    g : Quiver.Hom S X₂
    x₁ : X₁
    x₂ : X₂
    ⊢ Iff (CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl x₁) (Sum.inr x₂)) …
  -/
  constructor
    /-
      case mp
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      x₁ : X₁
      x₂ : X₂
      ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl x₁) (Sum.inr x₂) → Exi …
    -/
  · rintro ⟨_⟩
    /-
      case mp.inl_inr
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      s✝ : S
      ⊢ Exists fun s => And (Eq (f s✝) (f s)) (Eq (g s✝) (g s))
    -/
    exact ⟨_, rfl, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      x₁ : X₁
      x₂ : X₂
      ⊢ (Exists fun s => And (Eq x₁ (f s)) (Eq x₂ (g s))) → CategoryTheory.Limits.Ty …
    -/
  · rintro ⟨s, rfl, rfl⟩
    /-
      case mpr.intro.intro
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      s : S
      ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f s)) (Sum.inr (g s))
    -/
    exact Rel'.inl_inr _
    /-
      🎉 no goals
    -/


@[simp]
lemma inr_rel'_inr_iff (x₂ y₂ : X₂) :
    Rel' f g (Sum.inr x₂) (Sum.inr y₂) ↔ x₂ = y₂ := by
  /-
    S X₁ X₂ : Type u
    f : Quiver.Hom S X₁
    g : Quiver.Hom S X₂
    x₂ y₂ : X₂
    ⊢ Iff (CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr x₂) (Sum.inr y₂)) …
  -/
  constructor
    /-
      case mp
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      x₂ y₂ : X₂
      ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr x₂) (Sum.inr y₂) → Eq  …
    -/
  · rintro ⟨_⟩
    /-
      case mp.refl
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      x₂ : X₂
      ⊢ Eq x₂ x₂
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case mpr
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      x₂ y₂ : X₂
      ⊢ Eq x₂ y₂ → CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr x₂) (Sum.in …
    -/
  · rintro rfl
    /-
      case mpr
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      x₂ : X₂
      ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr x₂) (Sum.inr x₂)
    -/
    apply Rel'.refl
    /-
      🎉 no goals
    -/


lemma Rel'.symm {x y : X₁ ⊕ X₂} (h : Rel' f g x y) :
    Rel' f g y x := by
  /-
    S X₁ X₂ : Type u
    f : Quiver.Hom S X₁
    g : Quiver.Hom S X₂
    x y : Sum X₁ X₂
    h : CategoryTheory.Limits.Types.Pushout.Rel' f g x y
    ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g y x
  -/
  obtain _|⟨_, _, h⟩|_|_ := h
    /-
      case refl
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      x : Sum X₁ X₂
      ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g x x
    -/
  · apply Rel'.refl
    /-
      🎉 no goals
    -/
    /-
      case inl_inl
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      x₀✝ y₀✝ : S
      h : Eq (g x₀✝) (g y₀✝)
      ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f y₀✝)) (Sum.inl (f x …
    -/
  · exact Rel'.inl_inl _ _ h.symm
    /-
      🎉 no goals
    -/
    /-
      case inl_inr
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      s✝ : S
      ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr (g s✝)) (Sum.inl (f s✝))
    -/
  · exact Rel'.inr_inl _
    /-
      🎉 no goals
    -/
    /-
      case inr_inl
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      s✝ : S
      ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f s✝)) (Sum.inr (g s✝))
    -/
  · exact Rel'.inl_inr _
    /-
      🎉 no goals
    -/


lemma equivalence_rel' [Mono f] : _root_.Equivalence (Rel' f g) where
  refl := Rel'.refl
  symm h := h.symm
  trans := by
    /-
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      inst✝ : CategoryTheory.Mono f
      ⊢ ∀ {x y z : Sum X₁ X₂}, CategoryTheory.Limits.Types.Pushout.Rel' f g x y → Ca …
    -/
    rintro x y z (_|⟨_, _, h⟩|s|_) hyz
      /-
        case refl
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        inst✝ : CategoryTheory.Mono f
        x z : Sum X₁ X₂
        hyz : CategoryTheory.Limits.Types.Pushout.Rel' f g x z
        ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g x z
      -/
    · exact hyz
      /-
        🎉 no goals
      -/
      /-
        case inl_inl
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        inst✝ : CategoryTheory.Mono f
        z : Sum X₁ X₂
        x₀✝ y₀✝ : S
        h : Eq (g x₀✝) (g y₀✝)
        hyz : CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f y₀✝)) z
        ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f x₀✝)) z
      -/
    · obtain z₁|z₂ := z
        /-
          case inl_inl.inl
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          x₀✝ y₀✝ : S
          h : Eq (g x₀✝) (g y₀✝)
          z₁ : X₁
          hyz : CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f y₀✝)) (Sum.inl  …
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f x₀✝)) (Sum.inl z₁)
        -/
      · rw [inl_rel'_inl_iff] at hyz
        /-
          case inl_inl.inl
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          x₀✝ y₀✝ : S
          h : Eq (g x₀✝) (g y₀✝)
          z₁ : X₁
          hyz : Or (Eq (f y₀✝) z₁) (Exists fun x₀ => Exists fun y₀ => Exists fun x => An …
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f x₀✝)) (Sum.inl z₁)
        -/
        obtain rfl|⟨_, _, h', h'', rfl⟩ := hyz
          /-
            case inl_inl.inl.inl
            S X₁ X₂ : Type u
            f : Quiver.Hom S X₁
            g : Quiver.Hom S X₂
            inst✝ : CategoryTheory.Mono f
            x₀✝ y₀✝ : S
            h : Eq (g x₀✝) (g y₀✝)
            ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f x₀✝)) (Sum.inl (f y …
          -/
        · exact Rel'.inl_inl _ _ h
          /-
            🎉 no goals
          -/
          /-
            case inl_inl.inl.inr.intro.intro.intro.intro
            S X₁ X₂ : Type u
            f : Quiver.Hom S X₁
            g : Quiver.Hom S X₂
            inst✝ : CategoryTheory.Mono f
            x₀✝ y₀✝ : S
            h : Eq (g x₀✝) (g y₀✝)
            w✝¹ w✝ : S
            h' : Eq (g w✝¹) (g w✝)
            h'' : Eq (f y₀✝) (f w✝¹)
            ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f x₀✝)) (Sum.inl (f w …
          -/
        · obtain rfl := (mono_iff_injective f).1 inferInstance h''
          /-
            case inl_inl.inl.inr.intro.intro.intro.intro
            S X₁ X₂ : Type u
            f : Quiver.Hom S X₁
            g : Quiver.Hom S X₂
            inst✝ : CategoryTheory.Mono f
            x₀✝ y₀✝ : S
            h : Eq (g x₀✝) (g y₀✝)
            w✝ : S
            h' : Eq (g y₀✝) (g w✝)
            h'' : Eq (f y₀✝) (f y₀✝)
            ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f x₀✝)) (Sum.inl (f w …
          -/
          exact Rel'.inl_inl _ _ (h.trans h')
          /-
            🎉 no goals
          -/
        /-
          case inl_inl.inr
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          x₀✝ y₀✝ : S
          h : Eq (g x₀✝) (g y₀✝)
          z₂ : X₂
          hyz : CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f y₀✝)) (Sum.inr  …
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f x₀✝)) (Sum.inr z₂)
        -/
      · rw [inl_rel'_inr_iff] at hyz
        /-
          case inl_inl.inr
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          x₀✝ y₀✝ : S
          h : Eq (g x₀✝) (g y₀✝)
          z₂ : X₂
          hyz : Exists fun s => And (Eq (f y₀✝) (f s)) (Eq z₂ (g s))
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f x₀✝)) (Sum.inr z₂)
        -/
        obtain ⟨s, hs, rfl⟩ := hyz
        /-
          case inl_inl.inr.intro.intro
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          x₀✝ y₀✝ : S
          h : Eq (g x₀✝) (g y₀✝)
          s : S
          hs : Eq (f y₀✝) (f s)
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f x₀✝)) (Sum.inr (g s))
        -/
        obtain rfl := (mono_iff_injective f).1 inferInstance hs
        /-
          case inl_inl.inr.intro.intro
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          x₀✝ y₀✝ : S
          h : Eq (g x₀✝) (g y₀✝)
          hs : Eq (f y₀✝) (f y₀✝)
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f x₀✝)) (Sum.inr (g y …
        -/
        rw [← h]
        /-
          case inl_inl.inr.intro.intro
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          x₀✝ y₀✝ : S
          h : Eq (g x₀✝) (g y₀✝)
          hs : Eq (f y₀✝) (f y₀✝)
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f x₀✝)) (Sum.inr (g x …
        -/
        apply Rel'.inl_inr
        /-
          🎉 no goals
        -/
      /-
        case inl_inr
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        inst✝ : CategoryTheory.Mono f
        z : Sum X₁ X₂
        s : S
        hyz : CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr (g s)) z
        ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f s)) z
      -/
    · obtain z₁|z₂ := z
        /-
          case inl_inr.inl
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          s : S
          z₁ : X₁
          hyz : CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr (g s)) (Sum.inl z₁)
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f s)) (Sum.inl z₁)
        -/
      · replace hyz := hyz.symm
        /-
          case inl_inr.inl
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          s : S
          z₁ : X₁
          hyz : CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl z₁) (Sum.inr (g s))
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f s)) (Sum.inl z₁)
        -/
        rw [inl_rel'_inr_iff] at hyz
        /-
          case inl_inr.inl
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          s : S
          z₁ : X₁
          hyz : Exists fun s_1 => And (Eq z₁ (f s_1)) (Eq (g s) (g s_1))
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f s)) (Sum.inl z₁)
        -/
        obtain ⟨s', rfl, hs'⟩ := hyz
        /-
          case inl_inr.inl.intro.intro
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          s s' : S
          hs' : Eq (g s) (g s')
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f s)) (Sum.inl (f s'))
        -/
        exact Rel'.inl_inl _ _ hs'
        /-
          🎉 no goals
        -/
        /-
          case inl_inr.inr
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          s : S
          z₂ : X₂
          hyz : CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr (g s)) (Sum.inr z₂)
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f s)) (Sum.inr z₂)
        -/
      · rw [inr_rel'_inr_iff] at hyz
        /-
          case inl_inr.inr
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          s : S
          z₂ : X₂
          hyz : Eq (g s) z₂
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f s)) (Sum.inr z₂)
        -/
        subst hyz
        /-
          case inl_inr.inr
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          s : S
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f s)) (Sum.inr (g s))
        -/
        apply Rel'.inl_inr
        /-
          🎉 no goals
        -/
      /-
        case inr_inl
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        inst✝ : CategoryTheory.Mono f
        z : Sum X₁ X₂
        s✝ : S
        hyz : CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f s✝)) z
        ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr (g s✝)) z
      -/
    · obtain z₁|z₂ := z
        /-
          case inr_inl.inl
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          s✝ : S
          z₁ : X₁
          hyz : CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f s✝)) (Sum.inl z₁)
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr (g s✝)) (Sum.inl z₁)
        -/
      · rw [inl_rel'_inl_iff] at hyz
        /-
          case inr_inl.inl
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          s✝ : S
          z₁ : X₁
          hyz : Or (Eq (f s✝) z₁) (Exists fun x₀ => Exists fun y₀ => Exists fun x => And …
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr (g s✝)) (Sum.inl z₁)
        -/
        obtain rfl|⟨_, _, h, h', rfl⟩  := hyz
          /-
            case inr_inl.inl.inl
            S X₁ X₂ : Type u
            f : Quiver.Hom S X₁
            g : Quiver.Hom S X₂
            inst✝ : CategoryTheory.Mono f
            s✝ : S
            ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr (g s✝)) (Sum.inl (f s✝))
          -/
        · apply Rel'.inr_inl
          /-
            🎉 no goals
          -/
          /-
            case inr_inl.inl.inr.intro.intro.intro.intro
            S X₁ X₂ : Type u
            f : Quiver.Hom S X₁
            g : Quiver.Hom S X₂
            inst✝ : CategoryTheory.Mono f
            s✝ w✝¹ w✝ : S
            h : Eq (g w✝¹) (g w✝)
            h' : Eq (f s✝) (f w✝¹)
            ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr (g s✝)) (Sum.inl (f w✝))
          -/
        · obtain rfl := (mono_iff_injective f).1 inferInstance h'
          /-
            case inr_inl.inl.inr.intro.intro.intro.intro
            S X₁ X₂ : Type u
            f : Quiver.Hom S X₁
            g : Quiver.Hom S X₂
            inst✝ : CategoryTheory.Mono f
            s✝ w✝ : S
            h : Eq (g s✝) (g w✝)
            h' : Eq (f s✝) (f s✝)
            ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr (g s✝)) (Sum.inl (f w✝))
          -/
          rw [h]
          /-
            case inr_inl.inl.inr.intro.intro.intro.intro
            S X₁ X₂ : Type u
            f : Quiver.Hom S X₁
            g : Quiver.Hom S X₂
            inst✝ : CategoryTheory.Mono f
            s✝ w✝ : S
            h : Eq (g s✝) (g w✝)
            h' : Eq (f s✝) (f s✝)
            ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr (g w✝)) (Sum.inl (f w✝))
          -/
          apply Rel'.inr_inl
          /-
            🎉 no goals
          -/
        /-
          case inr_inl.inr
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          s✝ : S
          z₂ : X₂
          hyz : CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f s✝)) (Sum.inr z₂)
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr (g s✝)) (Sum.inr z₂)
        -/
      · rw [inl_rel'_inr_iff] at hyz
        /-
          case inr_inl.inr
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          s✝ : S
          z₂ : X₂
          hyz : Exists fun s => And (Eq (f s✝) (f s)) (Eq z₂ (g s))
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr (g s✝)) (Sum.inr z₂)
        -/
        obtain ⟨s, hs, rfl⟩ := hyz
        /-
          case inr_inl.inr.intro.intro
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          s✝ s : S
          hs : Eq (f s✝) (f s)
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr (g s✝)) (Sum.inr (g s))
        -/
        obtain rfl := (mono_iff_injective f).1 inferInstance hs
        /-
          case inr_inl.inr.intro.intro
          S X₁ X₂ : Type u
          f : Quiver.Hom S X₁
          g : Quiver.Hom S X₂
          inst✝ : CategoryTheory.Mono f
          s✝ : S
          hs : Eq (f s✝) (f s✝)
          ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inr (g s✝)) (Sum.inr (g s✝))
        -/
        apply Rel'.refl
        /-
          🎉 no goals
        -/


/-- The obvious equivalence `Pushout f g ≃ Pushout' f g`. -/
def equivPushout' : Pushout f g ≃ Pushout' f g where
  toFun := Quot.lift (Quot.mk _) (by
    /-
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      ⊢ ∀ (a b : Sum X₁ X₂), CategoryTheory.Limits.Types.Pushout.Rel f g a b → Eq (Q …
    -/
    rintro _ _ ⟨⟩
    /-
      case inl_inr
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      s✝ : S
      ⊢ Eq (Quot.mk (CategoryTheory.Limits.Types.Pushout.Rel' f g) (Sum.inl (f s✝))) …
    -/
    apply Quot.sound
    /-
      case inl_inr.a
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      s✝ : S
      ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f s✝)) (Sum.inr (g s✝))
    -/
    apply Rel'.inl_inr)
    /-
      🎉 no goals
    -/
  invFun := Quot.lift (Quot.mk _) (by
    /-
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      ⊢ ∀ (a b : Sum X₁ X₂), CategoryTheory.Limits.Types.Pushout.Rel' f g a b → Eq ( …
    -/
    rintro a b (_|⟨x₀, y₀, h⟩|_|_)
      /-
        case refl
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        a : Sum X₁ X₂
        ⊢ Eq (Quot.mk (CategoryTheory.Limits.Types.Pushout.Rel f g) a) (Quot.mk (Categ …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case inl_inl
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        x₀ y₀ : S
        h : Eq (g x₀) (g y₀)
        ⊢ Eq (Quot.mk (CategoryTheory.Limits.Types.Pushout.Rel f g) (Sum.inl (f x₀)))  …
      -/
    · have h₀ : Rel f g _ _ := Rel.inl_inr x₀
      /-
        case inl_inl
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        x₀ y₀ : S
        h : Eq (g x₀) (g y₀)
        h₀ : CategoryTheory.Limits.Types.Pushout.Rel f g (Sum.inl (f x₀)) (Sum.inr (g  …
        ⊢ Eq (Quot.mk (CategoryTheory.Limits.Types.Pushout.Rel f g) (Sum.inl (f x₀)))  …
      -/
      rw [Quot.sound h₀, h]
      /-
        case inl_inl
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        x₀ y₀ : S
        h : Eq (g x₀) (g y₀)
        h₀ : CategoryTheory.Limits.Types.Pushout.Rel f g (Sum.inl (f x₀)) (Sum.inr (g  …
        ⊢ Eq (Quot.mk (CategoryTheory.Limits.Types.Pushout.Rel f g) (Sum.inr (g y₀)))  …
      -/
      symm
      /-
        case inl_inl
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        x₀ y₀ : S
        h : Eq (g x₀) (g y₀)
        h₀ : CategoryTheory.Limits.Types.Pushout.Rel f g (Sum.inl (f x₀)) (Sum.inr (g  …
        ⊢ Eq (Quot.mk (CategoryTheory.Limits.Types.Pushout.Rel f g) (Sum.inl (f y₀)))  …
      -/
      apply Quot.sound
      /-
        case inl_inl.a
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        x₀ y₀ : S
        h : Eq (g x₀) (g y₀)
        h₀ : CategoryTheory.Limits.Types.Pushout.Rel f g (Sum.inl (f x₀)) (Sum.inr (g  …
        ⊢ CategoryTheory.Limits.Types.Pushout.Rel f g (Sum.inl (f y₀)) (Sum.inr (g y₀))
      -/
      apply Rel.inl_inr
      /-
        🎉 no goals
      -/
      /-
        case inl_inr
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        s✝ : S
        ⊢ Eq (Quot.mk (CategoryTheory.Limits.Types.Pushout.Rel f g) (Sum.inl (f s✝)))  …
      -/
    · apply Quot.sound
      /-
        case inl_inr.a
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        s✝ : S
        ⊢ CategoryTheory.Limits.Types.Pushout.Rel f g (Sum.inl (f s✝)) (Sum.inr (g s✝))
      -/
      apply Rel.inl_inr
      /-
        🎉 no goals
      -/
      /-
        case inr_inl
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        s✝ : S
        ⊢ Eq (Quot.mk (CategoryTheory.Limits.Types.Pushout.Rel f g) (Sum.inr (g s✝)))  …
      -/
    · symm
      /-
        case inr_inl
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        s✝ : S
        ⊢ Eq (Quot.mk (CategoryTheory.Limits.Types.Pushout.Rel f g) (Sum.inl (f s✝)))  …
      -/
      apply Quot.sound
      /-
        case inr_inl.a
        S X₁ X₂ : Type u
        f : Quiver.Hom S X₁
        g : Quiver.Hom S X₂
        s✝ : S
        ⊢ CategoryTheory.Limits.Types.Pushout.Rel f g (Sum.inl (f s✝)) (Sum.inr (g s✝))
      -/
      apply Rel.inl_inr)
      /-
        🎉 no goals
      -/
                 /-
                   S X₁ X₂ : Type u
                   f : Quiver.Hom S X₁
                   g : Quiver.Hom S X₂
                   ⊢ Function.LeftInverse (Quot.lift (Quot.mk (CategoryTheory.Limits.Types.Pushou …
                 -/
  left_inv := by rintro ⟨x⟩; rfl
                             /-
                               🎉 no goals
                             -/
                  /-
                    S X₁ X₂ : Type u
                    f : Quiver.Hom S X₁
                    g : Quiver.Hom S X₂
                    ⊢ Function.RightInverse (Quot.lift (Quot.mk (CategoryTheory.Limits.Types.Pusho …
                  -/
  right_inv := by rintro ⟨x⟩; rfl
                              /-
                                🎉 no goals
                              -/


lemma quot_mk_eq_iff [Mono f] (a b : X₁ ⊕ X₂) :
    (Quot.mk _ a : Pushout f g) = Quot.mk _ b ↔ Rel' f g a b := by
  /-
    S X₁ X₂ : Type u
    f : Quiver.Hom S X₁
    g : Quiver.Hom S X₂
    inst✝ : CategoryTheory.Mono f
    a b : Sum X₁ X₂
    ⊢ Iff (Eq (Quot.mk (CategoryTheory.Limits.Types.Pushout.Rel f g) a) (Quot.mk ( …
  -/
  rw [← (equivalence_rel' f g).quot_mk_eq_iff]
  exact ⟨fun h => (equivPushout' f g).symm.injective h,
    fun h => (equivPushout' f g).injective h⟩


lemma inl_eq_inr_iff [Mono f] (x₁ : X₁) (x₂ : X₂) :
    (inl f g x₁ = inr f g x₂) ↔
      ∃ (s : S), f s = x₁ ∧ g s = x₂ := by
  /-
    S X₁ X₂ : Type u
    f : Quiver.Hom S X₁
    g : Quiver.Hom S X₂
    inst✝ : CategoryTheory.Mono f
    x₁ : X₁
    x₂ : X₂
    ⊢ Iff (Eq (CategoryTheory.Limits.Types.Pushout.inl f g x₁) (CategoryTheory.Lim …
  -/
  refine (Pushout.quot_mk_eq_iff f g (Sum.inl x₁) (Sum.inr x₂)).trans ?_
  /-
    S X₁ X₂ : Type u
    f : Quiver.Hom S X₁
    g : Quiver.Hom S X₂
    inst✝ : CategoryTheory.Mono f
    x₁ : X₁
    x₂ : X₂
    ⊢ Iff (CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl x₁) (Sum.inr x₂)) …
  -/
  constructor
    /-
      case mp
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      inst✝ : CategoryTheory.Mono f
      x₁ : X₁
      x₂ : X₂
      ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl x₁) (Sum.inr x₂) → Exi …
    -/
  · rintro ⟨⟩
    /-
      case mp.inl_inr
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      inst✝ : CategoryTheory.Mono f
      s✝ : S
      ⊢ Exists fun s => And (Eq (f s) (f s✝)) (Eq (g s) (g s✝))
    -/
    exact ⟨_, rfl, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      inst✝ : CategoryTheory.Mono f
      x₁ : X₁
      x₂ : X₂
      ⊢ (Exists fun s => And (Eq (f s) x₁) (Eq (g s) x₂)) → CategoryTheory.Limits.Ty …
    -/
  · rintro ⟨s, rfl, rfl⟩
    /-
      case mpr.intro.intro
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      inst✝ : CategoryTheory.Mono f
      s : S
      ⊢ CategoryTheory.Limits.Types.Pushout.Rel' f g (Sum.inl (f s)) (Sum.inr (g s))
    -/
    apply Rel'.inl_inr
    /-
      🎉 no goals
    -/


lemma pushoutCocone_inl_eq_inr_imp_of_iso {c c' : PushoutCocone f g} (e : c ≅ c')
    (x₁ : X₁) (x₂ : X₂) (h : c.inl x₁ = c.inr x₂) :
    c'.inl x₁ = c'.inr x₂ := by
  /-
    S X₁ X₂ : Type u
    f : Quiver.Hom S X₁
    g : Quiver.Hom S X₂
    c c' : CategoryTheory.Limits.PushoutCocone f g
    e : CategoryTheory.Iso c c'
    x₁ : X₁
    x₂ : X₂
    h : Eq (c.inl x₁) (c.inr x₂)
    ⊢ Eq (c'.inl x₁) (c'.inr x₂)
  -/
  convert congr_arg e.hom.hom h
    /-
      case h.e'_2
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      c c' : CategoryTheory.Limits.PushoutCocone f g
      e : CategoryTheory.Iso c c'
      x₁ : X₁
      x₂ : X₂
      h : Eq (c.inl x₁) (c.inr x₂)
      ⊢ Eq (c'.inl x₁) (e.hom.hom (c.inl x₁))
    -/
  · exact congr_fun (e.hom.w WalkingSpan.left).symm x₁
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      c c' : CategoryTheory.Limits.PushoutCocone f g
      e : CategoryTheory.Iso c c'
      x₁ : X₁
      x₂ : X₂
      h : Eq (c.inl x₁) (c.inr x₂)
      ⊢ Eq (c'.inr x₂) (e.hom.hom (c.inr x₂))
    -/
  · exact congr_fun (e.hom.w WalkingSpan.right).symm x₂
    /-
      🎉 no goals
    -/


lemma pushoutCocone_inl_eq_inr_iff_of_iso {c c' : PushoutCocone f g} (e : c ≅ c')
    (x₁ : X₁) (x₂ : X₂) :
    c.inl x₁ = c.inr x₂ ↔ c'.inl x₁ = c'.inr x₂ := by
  /-
    S X₁ X₂ : Type u
    f : Quiver.Hom S X₁
    g : Quiver.Hom S X₂
    c c' : CategoryTheory.Limits.PushoutCocone f g
    e : CategoryTheory.Iso c c'
    x₁ : X₁
    x₂ : X₂
    ⊢ Iff (Eq (c.inl x₁) (c.inr x₂)) (Eq (c'.inl x₁) (c'.inr x₂))
  -/
  constructor
    /-
      case mp
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      c c' : CategoryTheory.Limits.PushoutCocone f g
      e : CategoryTheory.Iso c c'
      x₁ : X₁
      x₂ : X₂
      ⊢ Eq (c.inl x₁) (c.inr x₂) → Eq (c'.inl x₁) (c'.inr x₂)
    -/
  · apply pushoutCocone_inl_eq_inr_imp_of_iso e
    /-
      🎉 no goals
    -/
    /-
      case mpr
      S X₁ X₂ : Type u
      f : Quiver.Hom S X₁
      g : Quiver.Hom S X₂
      c c' : CategoryTheory.Limits.PushoutCocone f g
      e : CategoryTheory.Iso c c'
      x₁ : X₁
      x₂ : X₂
      ⊢ Eq (c'.inl x₁) (c'.inr x₂) → Eq (c.inl x₁) (c.inr x₂)
    -/
  · apply pushoutCocone_inl_eq_inr_imp_of_iso e.symm
    /-
      🎉 no goals
    -/


lemma pushoutCocone_inl_eq_inr_iff_of_isColimit {c : PushoutCocone f g} (hc : IsColimit c)
    (h₁ : Function.Injective f) (x₁ : X₁) (x₂ : X₂) :
    c.inl x₁ = c.inr x₂ ↔ ∃ (s : S), f s = x₁ ∧ g s = x₂ := by
  rw [pushoutCocone_inl_eq_inr_iff_of_iso
    (Cocones.ext (IsColimit.coconePointUniqueUpToIso hc (Pushout.isColimitCocone f g))
    (by aesop_cat))]
  /-
    S X₁ X₂ : Type u
    f : Quiver.Hom S X₁
    g : Quiver.Hom S X₂
    c : CategoryTheory.Limits.PushoutCocone f g
    hc : CategoryTheory.Limits.IsColimit c
    h₁ : Function.Injective f
    x₁ : X₁
    x₂ : X₂
    ⊢ Iff (Eq ((CategoryTheory.Limits.Types.Pushout.cocone f g).inl x₁) ((Category …
  -/
  have := (mono_iff_injective f).2 h₁
  /-
    S X₁ X₂ : Type u
    f : Quiver.Hom S X₁
    g : Quiver.Hom S X₂
    c : CategoryTheory.Limits.PushoutCocone f g
    hc : CategoryTheory.Limits.IsColimit c
    h₁ : Function.Injective f
    x₁ : X₁
    x₂ : X₂
    this : CategoryTheory.Mono f
    ⊢ Iff (Eq ((CategoryTheory.Limits.Types.Pushout.cocone f g).inl x₁) ((Category …
  -/
  apply Pushout.inl_eq_inr_iff
  /-
    🎉 no goals
  -/


/-- Given `I : MulticospanIndex (Type u)`, this is a type which identifies
to the sections of the functor `I.multicospan`. -/
@[ext]
structure MulticospanIndex.sections where
  /-- The data of an element in `I.left i` for each `i : I.L`. -/
  val (i : I.L) : I.left i
  property (r : I.R) : I.fst r (val _) = I.snd r (val _)


/-- The bijection `I.sections ≃ I.multicospan.sections` when `I : MulticospanIndex (Type u)`
is a multiequalizer diagram in the category of types. -/
@[simps]
def MulticospanIndex.sectionsEquiv :
    I.sections ≃ I.multicospan.sections where
  toFun s :=
    { val := fun i ↦ match i with
        | .left i => s.val i
        | .right j => I.fst j (s.val _)
      property := by
        /-
          I : CategoryTheory.Limits.MulticospanIndex (Type u)
          s : I.sections
          ⊢ Membership.mem I.multicospan.sections fun i => CategoryTheory.Limits.Multico …
        -/
        rintro _ _ (_|_|r)
          /-
            case id
            I : CategoryTheory.Limits.MulticospanIndex (Type u)
            s : I.sections
            j✝ : CategoryTheory.Limits.WalkingMulticospan I.fstTo I.sndTo
            ⊢ Eq (I.multicospan.map (CategoryTheory.Limits.WalkingMulticospan.Hom.id j✝) ( …
          -/
        · rfl
          /-
            🎉 no goals
          -/
          /-
            case fst
            I : CategoryTheory.Limits.MulticospanIndex (Type u)
            s : I.sections
            b✝ : I.R
            ⊢ Eq (I.multicospan.map (CategoryTheory.Limits.WalkingMulticospan.Hom.fst b✝)  …
          -/
        · rfl
          /-
            🎉 no goals
          -/
          /-
            case snd
            I : CategoryTheory.Limits.MulticospanIndex (Type u)
            s : I.sections
            r : I.R
            ⊢ Eq (I.multicospan.map (CategoryTheory.Limits.WalkingMulticospan.Hom.snd r) ( …
          -/
        · exact (s.property r).symm }
          /-
            🎉 no goals
          -/
  invFun s :=
    { val := fun i ↦ s.val (.left i)
      property := fun r ↦ (s.property (.fst r)).trans (s.property (.snd r)).symm }
  left_inv _ := rfl
  right_inv s := by
    /-
      I : CategoryTheory.Limits.MulticospanIndex (Type u)
      s : ↑I.multicospan.sections
      ⊢ Eq ((fun s => ⟨fun i => CategoryTheory.Limits.MulticospanIndex.sectionsEquiv …
    -/
    ext (_|r)
      /-
        case a.h.left
        I : CategoryTheory.Limits.MulticospanIndex (Type u)
        s : ↑I.multicospan.sections
        a✝ : I.L
        ⊢ Eq (↑((fun s => ⟨fun i => CategoryTheory.Limits.MulticospanIndex.sectionsEqu …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case a.h.right
        I : CategoryTheory.Limits.MulticospanIndex (Type u)
        s : ↑I.multicospan.sections
        r : I.R
        ⊢ Eq (↑((fun s => ⟨fun i => CategoryTheory.Limits.MulticospanIndex.sectionsEqu …
      -/
    · exact s.property (.fst r)
      /-
        🎉 no goals
      -/


/-- Given a multiequalizer diagram `I : MulticospanIndex (Type u)` in the category of
types and `c` a multifork for `I`, this is the canonical map `c.pt → I.sections`. -/
@[simps]
def toSections (x : c.pt) : I.sections where
  val i := c.ι i x
  property r := congr_fun (c.condition r) x


lemma toSections_fac : I.sectionsEquiv.symm ∘ Types.sectionOfCone c = c.toSections := rfl


/-- A multifork `c : Multifork I` in the category of types is limit iff the
map `c.toSections : c.pt → I.sections` is a bijection. -/
lemma isLimit_types_iff : Nonempty (IsLimit c) ↔ Function.Bijective c.toSections := by
  /-
    I : CategoryTheory.Limits.MulticospanIndex (Type u)
    c : CategoryTheory.Limits.Multifork I
    ⊢ Iff (Nonempty (CategoryTheory.Limits.IsLimit c)) (Function.Bijective c.toSec …
  -/
  rw [Types.isLimit_iff_bijective_sectionOfCone, ← toSections_fac, EquivLike.comp_bijective]
  /-
    🎉 no goals
  -/


/-- The bijection `I.sections ≃ c.pt` when `c : Multifork I` is a limit multifork
in the category of types. -/
noncomputable def sectionsEquiv : I.sections ≃ c.pt :=
  (Equiv.ofBijective _ (c.isLimit_types_iff.1 ⟨hc⟩)).symm


@[simp]
lemma sectionsEquiv_symm_apply_val (x : c.pt) (i : I.L) :
    ((sectionsEquiv hc).symm x).val i = c.ι i x := rfl


@[simp]
lemma sectionsEquiv_apply_val (s : I.sections) (i : I.L) :
    c.ι i (sectionsEquiv hc s) = s.val i := by
  /-
    I : CategoryTheory.Limits.MulticospanIndex (Type u)
    c : CategoryTheory.Limits.Multifork I
    hc : CategoryTheory.Limits.IsLimit c
    s : I.sections
    i : I.L
    ⊢ Eq (c.ι i ((CategoryTheory.Limits.Multifork.IsLimit.sectionsEquiv hc) s)) (s …
  -/
  obtain ⟨x, rfl⟩ := (sectionsEquiv hc).symm.surjective s
  /-
    case intro
    I : CategoryTheory.Limits.MulticospanIndex (Type u)
    c : CategoryTheory.Limits.Multifork I
    hc : CategoryTheory.Limits.IsLimit c
    i : I.L
    x : c.pt
    ⊢ Eq (c.ι i ((CategoryTheory.Limits.Multifork.IsLimit.sectionsEquiv hc) ((Cate …
  -/
  simp
  /-
    🎉 no goals
  -/


