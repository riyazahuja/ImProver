/-- We say an object in the free monoidal category is in normal form if it is of the form
    `(((𝟙_ C) ⊗ X₁) ⊗ X₂) ⊗ ⋯`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]
inductive NormalMonoidalObject : Type u
  | unit : NormalMonoidalObject
  | tensor : NormalMonoidalObject → C → NormalMonoidalObject


local notation "F" => FreeMonoidalCategory


local notation "N" => Discrete ∘ NormalMonoidalObject


local infixr:10 " ⟶ᵐ " => Hom

-- Porting note: this was automatic in mathlib 3

instance (x y : N C) : Subsingleton (x ⟶ y) := Discrete.instSubsingletonDiscreteHom _ _


/-- Auxiliary definition for `inclusion`. -/
@[simp]
def inclusionObj : NormalMonoidalObject C → F C
  | NormalMonoidalObject.unit => unit
  | NormalMonoidalObject.tensor n a => tensor (inclusionObj n) (of a)


/-- The discrete subcategory of objects in normal form includes into the free monoidal category. -/
def inclusion : N C ⥤ F C :=
  Discrete.functor inclusionObj


@[simp]
theorem inclusion_obj (X : N C) :
    inclusion.obj X = inclusionObj X.as :=
  rfl


@[simp]
theorem inclusion_map {X Y : N C} (f : X ⟶ Y) :
    inclusion.map f = eqToHom (congr_arg _ (Discrete.ext (Discrete.eq_of_hom f))) := by
  /-
    C : Type u
    X Y : Function.comp CategoryTheory.Discrete CategoryTheory.FreeMonoidalCategor …
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.FreeMonoidalCategory.inclusion.map f) (CategoryTheory.eqT …
  -/
  rcases f with ⟨⟨⟩⟩
  /-
    case up.up
    C : Type u
    X Y : Function.comp CategoryTheory.Discrete CategoryTheory.FreeMonoidalCategor …
    down✝ : Eq X.as Y.as
    ⊢ Eq (CategoryTheory.FreeMonoidalCategory.inclusion.map { down := { down := do …
  -/
  cases Discrete.ext (by assumption)
  /-
    case up.up.refl
    C : Type u
    X : Function.comp CategoryTheory.Discrete CategoryTheory.FreeMonoidalCategory. …
    down✝ : Eq X.as X.as
    ⊢ Eq (CategoryTheory.FreeMonoidalCategory.inclusion.map { down := { down := do …
  -/
  apply inclusion.map_id
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `normalize`. -/
def normalizeObj : F C → NormalMonoidalObject C → NormalMonoidalObject C
  | unit, n => n
  | of X, n => NormalMonoidalObject.tensor n X
  | tensor X Y, n => normalizeObj Y (normalizeObj X n)


@[simp]
theorem normalizeObj_unitor (n : NormalMonoidalObject C) : normalizeObj (𝟙_ (F C)) n = n :=
  rfl


@[simp]
theorem normalizeObj_tensor (X Y : F C) (n : NormalMonoidalObject C) :
    normalizeObj (X ⊗ Y) n = normalizeObj Y (normalizeObj X n) :=
  rfl


/-- Auxiliary definition for `normalize`. -/
def normalizeObj' (X : F C) : N C ⥤ N C := Discrete.functor fun n ↦ ⟨normalizeObj X n⟩


/-- Auxiliary definition for `normalize`. Here we prove that objects that are related by
    associators and unitors map to the same normal form. -/
@[simp]
def normalizeMapAux : ∀ {X Y : F C}, (X ⟶ᵐ Y) → (normalizeObj' X ⟶ normalizeObj' Y)
  | _, _, Hom.id _ => 𝟙 _
                            /-
                              C : Type u
                              X Y Z : CategoryTheory.FreeMonoidalCategory C
                              ⊢ Quiver.Hom ((X.tensor Y).tensor Z).normalizeObj' (X.tensor (Y.tensor Z)).nor …
                            -/
  | _, _, α_hom X Y Z => by dsimp; exact Discrete.natTrans (fun _ => 𝟙 _)
                                   /-
                                     🎉 no goals
                                   -/
                            /-
                              C : Type u
                              X✝ Y✝ Z✝ : CategoryTheory.FreeMonoidalCategory C
                              ⊢ Quiver.Hom (X✝.tensor (Y✝.tensor Z✝)).normalizeObj' ((X✝.tensor Y✝).tensor Z …
                            -/
  | _, _, α_inv _ _ _ => by dsimp; exact Discrete.natTrans (fun _ => 𝟙 _)
                                   /-
                                     🎉 no goals
                                   -/
                        /-
                          C : Type u
                          a✝ : CategoryTheory.FreeMonoidalCategory C
                          ⊢ Quiver.Hom (CategoryTheory.FreeMonoidalCategory.unit.tensor a✝).normalizeObj …
                        -/
  | _, _, l_hom _ => by dsimp; exact Discrete.natTrans (fun _ => 𝟙 _)
                               /-
                                 🎉 no goals
                               -/
                        /-
                          C : Type u
                          a✝ : CategoryTheory.FreeMonoidalCategory C
                          ⊢ Quiver.Hom a✝.normalizeObj' (CategoryTheory.FreeMonoidalCategory.unit.tensor …
                        -/
  | _, _, l_inv _ => by dsimp; exact Discrete.natTrans (fun _ => 𝟙 _)
                               /-
                                 🎉 no goals
                               -/
                        /-
                          C : Type u
                          a✝ : CategoryTheory.FreeMonoidalCategory C
                          ⊢ Quiver.Hom (a✝.tensor CategoryTheory.FreeMonoidalCategory.unit).normalizeObj …
                        -/
  | _, _, ρ_hom _ => by dsimp; exact Discrete.natTrans (fun _ => 𝟙 _)
                               /-
                                 🎉 no goals
                               -/
                        /-
                          C : Type u
                          a✝ : CategoryTheory.FreeMonoidalCategory C
                          ⊢ Quiver.Hom a✝.normalizeObj' (a✝.tensor CategoryTheory.FreeMonoidalCategory.u …
                        -/
  | _, _, ρ_inv _ => by dsimp; exact Discrete.natTrans (fun _ => 𝟙 _)
                               /-
                                 🎉 no goals
                               -/
  | _, _, (@comp _ _ _ _ f g) => normalizeMapAux f ≫ normalizeMapAux g
  | _, _, (@Hom.tensor _ T _ _ W f g) =>
    Discrete.natTrans <| fun ⟨X⟩ => (normalizeMapAux g).app ⟨normalizeObj T X⟩ ≫
      (normalizeObj' W).map ((normalizeMapAux f).app ⟨X⟩)
  | _, _, (@Hom.whiskerLeft _ T _ W f) =>
    Discrete.natTrans <| fun ⟨X⟩ => (normalizeMapAux f).app ⟨normalizeObj T X⟩
  | _, _, (@Hom.whiskerRight _ T _ f W) =>
    Discrete.natTrans <| fun X => (normalizeObj' W).map <| (normalizeMapAux f).app X


/-- Our normalization procedure works by first defining a functor `F C ⥤ (N C ⥤ N C)` (which turns
    out to be very easy), and then obtain a functor `F C ⥤ N C` by plugging in the normal object
    `𝟙_ C`. -/
@[simp]
def normalize : F C ⥤ N C ⥤ N C where
  obj X := normalizeObj' X
                                                 /-
                                                   C : Type u
                                                   X Y : CategoryTheory.FreeMonoidalCategory C
                                                   ⊢ ∀ (a b : X.Hom Y), HasEquiv.Equiv a b → Eq (CategoryTheory.FreeMonoidalCateg …
                                                 -/
  map {X Y} := Quotient.lift normalizeMapAux (by aesop_cat)
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- A variant of the normalization functor where we consider the result as an object in the free
    monoidal category (rather than an object of the discrete subcategory of objects in normal
    form). -/
@[simp]
def normalize' : F C ⥤ N C ⥤ F C :=
  normalize C ⋙ (whiskeringRight _ _ _).obj inclusion


/-- The normalization functor for the free monoidal category over `C`. -/
def fullNormalize : F C ⥤ N C where
  obj X := ((normalize C).obj X).obj ⟨NormalMonoidalObject.unit⟩
  map f := ((normalize C).map f).app ⟨NormalMonoidalObject.unit⟩


/-- Given an object `X` of the free monoidal category and an object `n` in normal form, taking
    the tensor product `n ⊗ X` in the free monoidal category is functorial in both `X` and `n`. -/
@[simp]
def tensorFunc : F C ⥤ N C ⥤ F C where
  obj X := Discrete.functor fun n => inclusion.obj ⟨n⟩ ⊗ X
  map f := Discrete.natTrans (fun _ => _ ◁ f)


theorem tensorFunc_map_app {X Y : F C} (f : X ⟶ Y) (n) : ((tensorFunc C).map f).app n = _ ◁ f :=
  rfl


theorem tensorFunc_obj_map (Z : F C) {n n' : N C} (f : n ⟶ n') :
    ((tensorFunc C).obj Z).map f = inclusion.map f ▷ Z := by
  /-
    C : Type u
    Z : CategoryTheory.FreeMonoidalCategory C
    n n' : Function.comp CategoryTheory.Discrete CategoryTheory.FreeMonoidalCatego …
    f : Quiver.Hom n n'
    ⊢ Eq (((CategoryTheory.FreeMonoidalCategory.tensorFunc C).obj Z).map f) (Categ …
  -/
  cases n
  /-
    case mk
    C : Type u
    Z : CategoryTheory.FreeMonoidalCategory C
    n' : Function.comp CategoryTheory.Discrete CategoryTheory.FreeMonoidalCategory …
    as✝ : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C
    f : Quiver.Hom { as := as✝ } n'
    ⊢ Eq (((CategoryTheory.FreeMonoidalCategory.tensorFunc C).obj Z).map f) (Categ …
  -/
  cases n'
  /-
    case mk.mk
    C : Type u
    Z : CategoryTheory.FreeMonoidalCategory C
    as✝¹ as✝ : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C
    f : Quiver.Hom { as := as✝¹ } { as := as✝ }
    ⊢ Eq (((CategoryTheory.FreeMonoidalCategory.tensorFunc C).obj Z).map f) (Categ …
  -/
  rcases f with ⟨⟨h⟩⟩
  /-
    case mk.mk.up.up
    C : Type u
    Z : CategoryTheory.FreeMonoidalCategory C
    as✝¹ as✝ : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C
    h : Eq { as := as✝¹ }.as { as := as✝ }.as
    ⊢ Eq (((CategoryTheory.FreeMonoidalCategory.tensorFunc C).obj Z).map { down := …
  -/
  dsimp at h
  /-
    case mk.mk.up.up
    C : Type u
    Z : CategoryTheory.FreeMonoidalCategory C
    as✝¹ as✝ : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C
    h : Eq as✝¹ as✝
    ⊢ Eq (((CategoryTheory.FreeMonoidalCategory.tensorFunc C).obj Z).map { down := …
  -/
  subst h
  /-
    case mk.mk.up.up
    C : Type u
    Z : CategoryTheory.FreeMonoidalCategory C
    as✝ : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C
    ⊢ Eq (((CategoryTheory.FreeMonoidalCategory.tensorFunc C).obj Z).map { down := …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `normalizeIso`. Here we construct the isomorphism between
    `n ⊗ X` and `normalize X n`. -/
@[simp]
def normalizeIsoApp :
    ∀ (X : F C) (n : N C), ((tensorFunc C).obj X).obj n ≅ ((normalize' C).obj X).obj n
  | of _, _ => Iso.refl _
  | unit, _ => ρ_ _
  | tensor X a, n =>
    (α_ _ _ _).symm ≪≫ whiskerRightIso (normalizeIsoApp X n) a ≪≫ normalizeIsoApp _ _


/-- Almost non-definitionally equal to `normalizeIsoApp`, but has a better definitional property
in the proof of `normalize_naturality`. -/
@[simp]
def normalizeIsoApp' :
    ∀ (X : F C) (n : NormalMonoidalObject C), inclusionObj n ⊗ X ≅ inclusionObj (normalizeObj X n)
  | of _, _ => Iso.refl _
  | unit, _ => ρ_ _
  | tensor X Y, n =>
    (α_ _ _ _).symm ≪≫ whiskerRightIso (normalizeIsoApp' X n) Y ≪≫ normalizeIsoApp' _ _


theorem normalizeIsoApp_eq :
    ∀ (X : F C) (n : N C), normalizeIsoApp C X n = normalizeIsoApp' C X n.as
  | of _, _ => rfl
  | unit, _ => rfl
  | tensor X Y, n => by
      /-
        C : Type u
        X Y : CategoryTheory.FreeMonoidalCategory C
        n : Function.comp CategoryTheory.Discrete CategoryTheory.FreeMonoidalCategory. …
        ⊢ Eq (CategoryTheory.FreeMonoidalCategory.normalizeIsoApp C (X.tensor Y) n) (C …
      -/
      rw [normalizeIsoApp, normalizeIsoApp']
      /-
        C : Type u
        X Y : CategoryTheory.FreeMonoidalCategory C
        n : Function.comp CategoryTheory.Discrete CategoryTheory.FreeMonoidalCategory. …
        ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.associator (CategoryTheory.FreeMo …
      -/
      rw [normalizeIsoApp_eq X n]
      /-
        C : Type u
        X Y : CategoryTheory.FreeMonoidalCategory C
        n : Function.comp CategoryTheory.Discrete CategoryTheory.FreeMonoidalCategory. …
        ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.associator (CategoryTheory.FreeMo …
      -/
      rw [normalizeIsoApp_eq Y ⟨normalizeObj X n.as⟩]
      /-
        C : Type u
        X Y : CategoryTheory.FreeMonoidalCategory C
        n : Function.comp CategoryTheory.Discrete CategoryTheory.FreeMonoidalCategory. …
        ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.associator (CategoryTheory.FreeMo …
      -/
      rfl
      /-
        🎉 no goals
      -/


@[simp]
theorem normalizeIsoApp_tensor (X Y : F C) (n : N C) :
    normalizeIsoApp C (X ⊗ Y) n =
      (α_ _ _ _).symm ≪≫ whiskerRightIso (normalizeIsoApp C X n) Y ≪≫ normalizeIsoApp _ _ _ :=
  rfl


@[simp]
theorem normalizeIsoApp_unitor (n : N C) : normalizeIsoApp C (𝟙_ (F C)) n = ρ_ _ :=
  rfl


/-- Auxiliary definition for `normalizeIso`. -/
@[simp]
def normalizeIsoAux (X : F C) : (tensorFunc C).obj X ≅ (normalize' C).obj X :=
  NatIso.ofComponents (normalizeIsoApp C X)
    (by
      /-
        C : Type u
        X : CategoryTheory.FreeMonoidalCategory C
        ⊢ ∀ {X_1 Y : Function.comp CategoryTheory.Discrete CategoryTheory.FreeMonoidal …
      -/
      rintro ⟨X⟩ ⟨Y⟩ ⟨⟨f⟩⟩
      /-
        case mk.mk.up.up
        C : Type u
        X✝ : CategoryTheory.FreeMonoidalCategory C
        X Y : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C
        f : Eq { as := X }.as { as := Y }.as
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.FreeMonoidalCategor …
      -/
      dsimp at f
      /-
        case mk.mk.up.up
        C : Type u
        X✝ : CategoryTheory.FreeMonoidalCategory C
        X Y : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C
        f : Eq X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.FreeMonoidalCategor …
      -/
      subst f
      /-
        case mk.mk.up.up
        C : Type u
        X✝ : CategoryTheory.FreeMonoidalCategory C
        X : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.FreeMonoidalCategor …
      -/
      dsimp
      /-
        case mk.mk.up.up
        C : Type u
        X✝ : CategoryTheory.FreeMonoidalCategory C
        X : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Ca …
      -/
      simp)
      /-
        🎉 no goals
      -/



theorem normalizeObj_congr (n : NormalMonoidalObject C) {X Y : F C} (f : X ⟶ Y) :
    normalizeObj X n = normalizeObj Y n := by
  /-
    C : Type u
    n : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C
    X Y : CategoryTheory.FreeMonoidalCategory C
    f : Quiver.Hom X Y
    ⊢ Eq (X.normalizeObj n) (Y.normalizeObj n)
  -/
  rcases f with ⟨f'⟩
  /-
    case mk
    C : Type u
    n : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C
    X Y : CategoryTheory.FreeMonoidalCategory C
    f : Quiver.Hom X Y
    f' : X.Hom Y
    ⊢ Eq (X.normalizeObj n) (Y.normalizeObj n)
  -/
  apply @congr_fun _ _ fun n => normalizeObj X n
  /-
    case mk.h
    C : Type u
    n : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C
    X Y : CategoryTheory.FreeMonoidalCategory C
    f : Quiver.Hom X Y
    f' : X.Hom Y
    ⊢ Eq (fun n => X.normalizeObj n) Y.normalizeObj
  -/
  clear n f
  induction f' with
  | comp _ _ _ _ => apply Eq.trans <;> assumption
  | whiskerLeft  _ _ ih => funext; apply congr_fun ih
  | whiskerRight _ _ ih => funext; apply congr_arg₂ _ rfl (congr_fun ih _)
  | @tensor W X Y Z _ _ ih₁ ih₂ =>
      funext n
      simp [congr_fun ih₁ n, congr_fun ih₂ (normalizeObj Y n)]
  | _ => funext; rfl


theorem normalize_naturality (n : NormalMonoidalObject C) {X Y : F C} (f : X ⟶ Y) :
    inclusionObj n ◁ f ≫ (normalizeIsoApp' C Y n).hom =
      (normalizeIsoApp' C X n).hom ≫
        inclusion.map (eqToHom (Discrete.ext (normalizeObj_congr n f))) := by
  /-
    C : Type u
    n : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C
    X Y : CategoryTheory.FreeMonoidalCategory C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  revert n
  /-
    C : Type u
    X Y : CategoryTheory.FreeMonoidalCategory C
    f : Quiver.Hom X Y
    ⊢ ∀ (n : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C), Eq (Cate …
  -/
  induction f using Hom.inductionOn
  /-
    case id
    C : Type u
    X Y X✝ : CategoryTheory.FreeMonoidalCategory C
    ⊢ ∀ (n : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C), Eq (Cate …
  -/
  case comp f g ihf ihg => simp [ihg, reassoc_of% (ihf _)]
  case whiskerLeft X' X Y f ih =>
    intro n
    dsimp only [normalizeObj_tensor, normalizeIsoApp', tensor_eq_tensor, Iso.trans_hom,
      Iso.symm_hom, whiskerRightIso_hom, Function.comp_apply, inclusion_obj]
    rw [associator_inv_naturality_right_assoc, whisker_exchange_assoc, ih]
    simp
  case whiskerRight X Y h η' ih =>
    intro n
    dsimp only [normalizeObj_tensor, normalizeIsoApp', tensor_eq_tensor, Iso.trans_hom,
      Iso.symm_hom, whiskerRightIso_hom, Function.comp_apply, inclusion_obj]
    rw [associator_inv_naturality_middle_assoc, ← comp_whiskerRight_assoc, ih]
    have := dcongr_arg (fun x => (normalizeIsoApp' C η' x).hom) (normalizeObj_congr n h)
    simp [this]
  /-
    case id
    C : Type u
    X Y X✝ : CategoryTheory.FreeMonoidalCategory C
    ⊢ ∀ (n : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C), Eq (Cate …
  -/
  all_goals simp
  /-
    🎉 no goals
  -/


set_option tactic.skipAssignedInstances false in
/-- The isomorphism between `n ⊗ X` and `normalize X n` is natural (in both `X` and `n`, but
    naturality in `n` is trivial and was "proved" in `normalizeIsoAux`). This is the real heart
    of our proof of the coherence theorem. -/
def normalizeIso : tensorFunc C ≅ normalize' C :=
  NatIso.ofComponents (normalizeIsoAux C) <| by
    /-
      C : Type u
      ⊢ ∀ {X Y : CategoryTheory.FreeMonoidalCategory C} (f : Quiver.Hom X Y), Eq (Ca …
    -/
    intro X Y f
    /-
      C : Type u
      X Y : CategoryTheory.FreeMonoidalCategory C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.FreeMonoidalCategory …
    -/
    ext ⟨n⟩
    /-
      case w.h.mk
      C : Type u
      X Y : CategoryTheory.FreeMonoidalCategory C
      f : Quiver.Hom X Y
      n : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.FreeMonoidalCategor …
    -/
    convert normalize_naturality n f using 1
    /-
      case h.e'_2.h
      C : Type u
      X Y : CategoryTheory.FreeMonoidalCategory C
      f : Quiver.Hom X Y
      n : CategoryTheory.FreeMonoidalCategory.NormalMonoidalObject C
      e_1✝ : Eq (Quiver.Hom (((CategoryTheory.FreeMonoidalCategory.tensorFunc C).obj …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.FreeMonoidalCategor …
    -/
    any_goals dsimp [NatIso.ofComponents]; congr; apply normalizeIsoApp_eq
    /-
      🎉 no goals
    -/


/-- The isomorphism between an object and its normal form is natural. -/
def fullNormalizeIso : 𝟭 (F C) ≅ fullNormalize C ⋙ inclusion :=
  NatIso.ofComponents
  (fun X => (λ_ X).symm ≪≫ ((normalizeIso C).app X).app ⟨NormalMonoidalObject.unit⟩)
    (by
      /-
        C : Type u
        ⊢ ∀ {X Y : CategoryTheory.FreeMonoidalCategory C} (f : Quiver.Hom X Y), Eq (Ca …
      -/
      intro X Y f
      /-
        C : Type u
        X Y : CategoryTheory.FreeMonoidalCategory C
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Category …
      -/
      dsimp
      /-
        C : Type u
        X Y : CategoryTheory.FreeMonoidalCategory C
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
      -/
      rw [leftUnitor_inv_naturality_assoc, Category.assoc, Iso.cancel_iso_inv_left]
      exact
        congr_arg (fun f => NatTrans.app f (Discrete.mk NormalMonoidalObject.unit))
          ((normalizeIso.{u} C).hom.naturality f))


/-- The monoidal coherence theorem. -/
instance subsingleton_hom : Quiver.IsThin (F C) := fun X Y =>
  ⟨fun f g => by
    /-
      C : Type u
      X Y : CategoryTheory.FreeMonoidalCategory C
      f g : Quiver.Hom X Y
      ⊢ Eq f g
    -/
    have hfg : (fullNormalize C).map f = (fullNormalize C).map g := Subsingleton.elim _ _
    /-
      C : Type u
      X Y : CategoryTheory.FreeMonoidalCategory C
      f g : Quiver.Hom X Y
      hfg : Eq ((CategoryTheory.FreeMonoidalCategory.fullNormalize C).map f) ((Categ …
      ⊢ Eq f g
    -/
    have hf := NatIso.naturality_2 (fullNormalizeIso.{u} C) f
    /-
      C : Type u
      X Y : CategoryTheory.FreeMonoidalCategory C
      f g : Quiver.Hom X Y
      hfg : Eq ((CategoryTheory.FreeMonoidalCategory.fullNormalize C).map f) ((Categ …
      hf : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.FreeMonoidalCateg …
      ⊢ Eq f g
    -/
    have hg := NatIso.naturality_2 (fullNormalizeIso.{u} C) g
    /-
      C : Type u
      X Y : CategoryTheory.FreeMonoidalCategory C
      f g : Quiver.Hom X Y
      hfg : Eq ((CategoryTheory.FreeMonoidalCategory.fullNormalize C).map f) ((Categ …
      hf : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.FreeMonoidalCateg …
      hg : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.FreeMonoidalCateg …
      ⊢ Eq f g
    -/
    exact hf.symm.trans (Eq.trans (by simp only [Functor.comp_map, hfg]) hg)⟩
    /-
      🎉 no goals
    -/


/-- Auxiliary construction for showing that the free monoidal category is a groupoid. Do not use
    this, use `IsIso.inv` instead. -/
def inverseAux : ∀ {X Y : F C}, (X ⟶ᵐ Y) → (Y ⟶ᵐ X)
  | _, _, Hom.id X => id X
  | _, _, α_hom _ _ _ => α_inv _ _ _
  | _, _, α_inv _ _ _ => α_hom _ _ _
  | _, _, ρ_hom _ => ρ_inv _
  | _, _, ρ_inv _ => ρ_hom _
  | _, _, l_hom _ => l_inv _
  | _, _, l_inv _ => l_hom _
  | _, _, comp f g => (inverseAux g).comp (inverseAux f)
  | _, _, Hom.whiskerLeft X f => (inverseAux f).whiskerLeft X
  | _, _, Hom.whiskerRight f X => (inverseAux f).whiskerRight X
  | _, _, Hom.tensor f g => (inverseAux f).tensor (inverseAux g)


instance : Groupoid.{u} (F C) :=
  { (inferInstance : Category (F C)) with
                                                       /-
                                                         C : Type u
                                                         X✝ Y✝ : CategoryTheory.FreeMonoidalCategory C
                                                         ⊢ ∀ (a b : X✝.Hom Y✝), HasEquiv.Equiv a b → Eq ((fun f => Quotient.mk (Y✝.seto …
                                                       -/
    inv := Quotient.lift (fun f => ⟦inverseAux f⟧) (by aesop_cat) }
                                                       /-
                                                         🎉 no goals
                                                       -/


