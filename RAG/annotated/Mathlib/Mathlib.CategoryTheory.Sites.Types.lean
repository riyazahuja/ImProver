/-- A Grothendieck topology associated to the category of all types.
A sieve is a covering iff it is jointly surjective. -/
def typesGrothendieckTopology : GrothendieckTopology (Type u) where
  sieves α S := ∀ x : α, S fun _ : PUnit => x
  top_mem' _ _ := trivial
  pullback_stable' _ _ _ f hs x := hs (f x)
  transitive' _ _ hs _ hr x := hr (hs x) PUnit.unit


/-- The discrete sieve on a type, which only includes arrows whose image is a subsingleton. -/
@[simps]
def discreteSieve (α : Type u) : Sieve α where
  arrows _ f := ∃ x, ∀ y, f y = x
  downward_closed := fun ⟨x, hx⟩ g => ⟨x, fun y => hx <| g y⟩


theorem discreteSieve_mem (α : Type u) : discreteSieve α ∈ typesGrothendieckTopology α :=
  fun x => ⟨x, fun _ => rfl⟩


/-- The discrete presieve on a type, which only includes arrows whose domain is a singleton. -/
def discretePresieve (α : Type u) : Presieve α :=
  fun β _ => ∃ x : β, ∀ y : β, y = x


theorem generate_discretePresieve_mem (α : Type u) :
    Sieve.generate (discretePresieve α) ∈ typesGrothendieckTopology α :=
  fun x => ⟨PUnit, id, fun _ => x, ⟨PUnit.unit, fun _ => Subsingleton.elim _ _⟩, rfl⟩


/-- The sheaf condition for `yoneda'`. -/
theorem Presieve.isSheaf_yoneda' {α : Type u} :
    Presieve.IsSheaf typesGrothendieckTopology (yoneda.obj α) :=
  fun β _ hs x hx =>
  ⟨fun y => x _ (hs y) PUnit.unit, fun γ f h =>
    funext fun z => by
      /-
        α β : Type u
        x✝ : CategoryTheory.Sieve β
        hs : Membership.mem (CategoryTheory.typesGrothendieckTopology β) x✝
        x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj α) x✝. …
        hx : x.Compatible
        γ : Type u
        f : Quiver.Hom γ β
        h : x✝.arrows f
        z : Opposite.unop { unop := γ }
        ⊢ Eq ((CategoryTheory.yoneda.obj α).map f.op (fun y => x (fun x => y) ⋯ PUnit. …
      -/
      convert congr_fun (hx (𝟙 _) (fun _ => z) (hs <| f z) h rfl) PUnit.unit using 1,
      /-
        🎉 no goals
      -/
                                   /-
                                     α β : Type u
                                     x✝ : CategoryTheory.Sieve β
                                     hs : Membership.mem (CategoryTheory.typesGrothendieckTopology β) x✝
                                     x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.yoneda.obj α) x✝. …
                                     hx : x.Compatible
                                     f : (CategoryTheory.yoneda.obj α).obj { unop := β }
                                     hf : (fun t => x.IsAmalgamation t) f
                                     y : Opposite.unop { unop := β }
                                     ⊢ Eq (f y) (x (fun x => y) ⋯ PUnit.unit)
                                   -/
    fun f hf => funext fun y => by convert congr_fun (hf _ (hs y)) PUnit.unit⟩
                                   /-
                                     🎉 no goals
                                   -/


/-- The sheaf condition for `yoneda'`. -/
theorem Presheaf.isSheaf_yoneda' {α : Type u} :
    Presheaf.IsSheaf typesGrothendieckTopology (yoneda.obj α) := by
  /-
    α : Type u
    ⊢ CategoryTheory.Presheaf.IsSheaf CategoryTheory.typesGrothendieckTopology (Ca …
  -/
  rw [isSheaf_iff_isSheaf_of_type]
  /-
    α : Type u
    ⊢ CategoryTheory.Presieve.IsSheaf CategoryTheory.typesGrothendieckTopology (Ca …
  -/
  exact Presieve.isSheaf_yoneda'
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-26")] alias isSheaf_yoneda' := Presieve.isSheaf_yoneda'


/-- The yoneda functor that sends a type to a sheaf over the category of types. -/
@[simps]
def yoneda' : Type u ⥤ Sheaf typesGrothendieckTopology (Type u) where
  obj α := ⟨yoneda.obj α, Presheaf.isSheaf_yoneda'⟩
  map f := ⟨yoneda.map f⟩


@[simp]
theorem yoneda'_comp : yoneda'.{u} ⋙ sheafToPresheaf _ _ = yoneda :=
  rfl


/-- Given a presheaf `P` on the category of types, construct
a map `P(α) → (α → P(*))` for all type `α`. -/
def eval (P : Type uᵒᵖ ⥤ Type u) (α : Type u) (s : P.obj (op α)) (x : α) : P.obj (op PUnit) :=
  P.map (↾fun _ => x).op s


/-- Given a sheaf `S` on the category of types, construct a map
`(α → S(*)) → S(α)` that is inverse to `eval`. -/
noncomputable def typesGlue (S : Type uᵒᵖ ⥤ Type u) (hs : IsSheaf typesGrothendieckTopology S)
    (α : Type u) (f : α → S.obj (op PUnit)) : S.obj (op α) :=
  (hs.isSheafFor _ _ (generate_discretePresieve_mem α)).amalgamate
    (fun _ g hg => S.map (↾fun _ => PUnit.unit).op <| f <| g <| Classical.choose hg)
    fun β γ δ g₁ g₂ f₁ f₂ hf₁ hf₂ h =>
    (hs.isSheafFor _ _ (generate_discretePresieve_mem δ)).isSeparatedFor.ext fun ε g ⟨x, _⟩ => by
      have : f₁ (Classical.choose hf₁) = f₂ (Classical.choose hf₂) :=
        Classical.choose_spec hf₁ (g₁ <| g x) ▸
          Classical.choose_spec hf₂ (g₂ <| g x) ▸ congr_fun h _
      /-
        S : CategoryTheory.Functor (Opposite (Type u)) (Type u)
        hs : CategoryTheory.Presieve.IsSheaf CategoryTheory.typesGrothendieckTopology S
        α : Type u
        f : α → S.obj { unop := PUnit.{u + 1} }
        β γ δ : Type u
        g₁ : Quiver.Hom δ β
        g₂ : Quiver.Hom δ γ
        f₁ : Quiver.Hom β α
        f₂ : Quiver.Hom γ α
        hf₁ : CategoryTheory.discretePresieve α f₁
        hf₂ : CategoryTheory.discretePresieve α f₂
        h : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategoryStru …
        ε : Type u
        g : Quiver.Hom ε δ
        x✝ : CategoryTheory.discretePresieve δ g
        x : ε
        h✝ : ∀ (y : ε), Eq y x
        this : Eq (f₁ (Classical.choose hf₁)) (f₂ (Classical.choose hf₂))
        ⊢ Eq (S.map g.op (S.map g₁.op ((fun x g hg => S.map (CategoryTheory.asHom fun  …
      -/
      simp_rw [← FunctorToTypes.map_comp_apply, this, ← op_comp]
      /-
        S : CategoryTheory.Functor (Opposite (Type u)) (Type u)
        hs : CategoryTheory.Presieve.IsSheaf CategoryTheory.typesGrothendieckTopology S
        α : Type u
        f : α → S.obj { unop := PUnit.{u + 1} }
        β γ δ : Type u
        g₁ : Quiver.Hom δ β
        g₂ : Quiver.Hom δ γ
        f₁ : Quiver.Hom β α
        f₂ : Quiver.Hom γ α
        hf₁ : CategoryTheory.discretePresieve α f₁
        hf₂ : CategoryTheory.discretePresieve α f₂
        h : Eq (CategoryTheory.CategoryStruct.comp g₁ f₁) (CategoryTheory.CategoryStru …
        ε : Type u
        g : Quiver.Hom ε δ
        x✝ : CategoryTheory.discretePresieve δ g
        x : ε
        h✝ : ∀ (y : ε), Eq y x
        this : Eq (f₁ (Classical.choose hf₁)) (f₂ (Classical.choose hf₂))
        ⊢ Eq (S.map (CategoryTheory.CategoryStruct.comp g (CategoryTheory.CategoryStru …
      -/
      rfl
      /-
        🎉 no goals
      -/


theorem eval_typesGlue {S hs α} (f) : eval.{u} S α (typesGlue S hs α f) = f := by
  /-
    S : CategoryTheory.Functor (Opposite (Type u)) (Type u)
    hs : CategoryTheory.Presieve.IsSheaf CategoryTheory.typesGrothendieckTopology S
    α : Type u
    f : α → S.obj { unop := PUnit.{u + 1} }
    ⊢ Eq (CategoryTheory.eval S α (CategoryTheory.typesGlue S hs α f)) f
  -/
  funext x
  /-
    case h
    S : CategoryTheory.Functor (Opposite (Type u)) (Type u)
    hs : CategoryTheory.Presieve.IsSheaf CategoryTheory.typesGrothendieckTopology S
    α : Type u
    f : α → S.obj { unop := PUnit.{u + 1} }
    x : α
    ⊢ Eq (CategoryTheory.eval S α (CategoryTheory.typesGlue S hs α f) x) (f x)
  -/
  apply (IsSheafFor.valid_glue _ _ _ <| ⟨PUnit.unit, fun _ => Subsingleton.elim _ _⟩).trans
  /-
    case h
    S : CategoryTheory.Functor (Opposite (Type u)) (Type u)
    hs : CategoryTheory.Presieve.IsSheaf CategoryTheory.typesGrothendieckTopology S
    α : Type u
    f : α → S.obj { unop := PUnit.{u + 1} }
    x : α
    ⊢ Eq (S.map (CategoryTheory.asHom fun x => PUnit.unit).op (f (CategoryTheory.a …
  -/
  convert FunctorToTypes.map_id_apply S _
  /-
    🎉 no goals
  -/


theorem typesGlue_eval {S hs α} (s) : typesGlue.{u} S hs α (eval S α s) = s := by
  /-
    S : CategoryTheory.Functor (Opposite (Type u)) (Type u)
    hs : CategoryTheory.Presieve.IsSheaf CategoryTheory.typesGrothendieckTopology S
    α : Type u
    s : S.obj { unop := α }
    ⊢ Eq (CategoryTheory.typesGlue S hs α (CategoryTheory.eval S α s)) s
  -/
  apply (hs.isSheafFor _ _ (generate_discretePresieve_mem α)).isSeparatedFor.ext
  /-
    S : CategoryTheory.Functor (Opposite (Type u)) (Type u)
    hs : CategoryTheory.Presieve.IsSheaf CategoryTheory.typesGrothendieckTopology S
    α : Type u
    s : S.obj { unop := α }
    ⊢ ∀ ⦃Y : Type u⦄ ⦃f : Quiver.Hom Y α⦄, CategoryTheory.discretePresieve α f → E …
  -/
  intro β f hf
  /-
    S : CategoryTheory.Functor (Opposite (Type u)) (Type u)
    hs : CategoryTheory.Presieve.IsSheaf CategoryTheory.typesGrothendieckTopology S
    α : Type u
    s : S.obj { unop := α }
    β : Type u
    f : Quiver.Hom β α
    hf : CategoryTheory.discretePresieve α f
    ⊢ Eq (S.map f.op (CategoryTheory.typesGlue S hs α (CategoryTheory.eval S α s)) …
  -/
  apply (IsSheafFor.valid_glue _ _ _ hf).trans
  /-
    S : CategoryTheory.Functor (Opposite (Type u)) (Type u)
    hs : CategoryTheory.Presieve.IsSheaf CategoryTheory.typesGrothendieckTopology S
    α : Type u
    s : S.obj { unop := α }
    β : Type u
    f : Quiver.Hom β α
    hf : CategoryTheory.discretePresieve α f
    ⊢ Eq (S.map (CategoryTheory.asHom fun x => PUnit.unit).op (CategoryTheory.eval …
  -/
  apply (FunctorToTypes.map_comp_apply _ _ _ _).symm.trans
  /-
    S : CategoryTheory.Functor (Opposite (Type u)) (Type u)
    hs : CategoryTheory.Presieve.IsSheaf CategoryTheory.typesGrothendieckTopology S
    α : Type u
    s : S.obj { unop := α }
    β : Type u
    f : Quiver.Hom β α
    hf : CategoryTheory.discretePresieve α f
    ⊢ Eq (S.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.asHom fun x => …
  -/
  rw [← op_comp]
  --congr 2 -- Porting note: This tactic didn't work. Find an alternative.
  /-
    S : CategoryTheory.Functor (Opposite (Type u)) (Type u)
    hs : CategoryTheory.Presieve.IsSheaf CategoryTheory.typesGrothendieckTopology S
    α : Type u
    s : S.obj { unop := α }
    β : Type u
    f : Quiver.Hom β α
    hf : CategoryTheory.discretePresieve α f
    ⊢ Eq (S.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.asHom fun x => …
  -/
  suffices ((↾fun _ ↦ PUnit.unit) ≫ ↾fun _ ↦ f (Classical.choose hf)) = f by rw [this]
  /-
    S : CategoryTheory.Functor (Opposite (Type u)) (Type u)
    hs : CategoryTheory.Presieve.IsSheaf CategoryTheory.typesGrothendieckTopology S
    α : Type u
    s : S.obj { unop := α }
    β : Type u
    f : Quiver.Hom β α
    hf : CategoryTheory.discretePresieve α f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.asHom fun x => PUnit. …
  -/
  funext x
  /-
    case h
    S : CategoryTheory.Functor (Opposite (Type u)) (Type u)
    hs : CategoryTheory.Presieve.IsSheaf CategoryTheory.typesGrothendieckTopology S
    α : Type u
    s : S.obj { unop := α }
    β : Type u
    f : Quiver.Hom β α
    hf : CategoryTheory.discretePresieve α f
    x : β
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.asHom fun x => PUnit. …
  -/
  exact congr_arg f (Classical.choose_spec hf x).symm
  /-
    🎉 no goals
  -/


/-- Given a sheaf `S`, construct an equivalence `S(α) ≃ (α → S(*))`. -/
@[simps]
noncomputable def evalEquiv (S : Type uᵒᵖ ⥤ Type u)
    (hs : Presheaf.IsSheaf typesGrothendieckTopology S)
    (α : Type u) : S.obj (op α) ≃ (α → S.obj (op PUnit)) where
  toFun := eval S α
  invFun := typesGlue S ((isSheaf_iff_isSheaf_of_type _ _ ).1 hs) α
  left_inv := typesGlue_eval
  right_inv := eval_typesGlue


theorem eval_map (S : Type uᵒᵖ ⥤ Type u) (α β) (f : β ⟶ α) (s x) :
    eval S β (S.map f.op s) x = eval S α s (f x) := by
  /-
    S : CategoryTheory.Functor (Opposite (Type u)) (Type u)
    α β : Type u
    f : Quiver.Hom β α
    s : S.obj { unop := α }
    x : β
    ⊢ Eq (CategoryTheory.eval S β (S.map f.op s) x) (CategoryTheory.eval S α s (f  …
  -/
  simp_rw [eval, ← FunctorToTypes.map_comp_apply, ← op_comp]; rfl
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- Given a sheaf `S`, construct an isomorphism `S ≅ [-, S(*)]`. -/
@[simps!]
noncomputable def equivYoneda (S : Type uᵒᵖ ⥤ Type u)
    (hs : Presheaf.IsSheaf typesGrothendieckTopology S) :
    S ≅ yoneda.obj (S.obj (op PUnit)) :=
  NatIso.ofComponents (fun α => Equiv.toIso <| evalEquiv S hs <| unop α) fun {α β} f =>
    funext fun _ => funext fun _ => eval_map S (unop α) (unop β) f.unop _ _


/-- Given a sheaf `S`, construct an isomorphism `S ≅ [-, S(*)]`. -/
@[simps]
noncomputable def equivYoneda' (S : Sheaf typesGrothendieckTopology (Type u)) :
    S ≅ yoneda'.obj (S.1.obj (op PUnit)) where
  hom := ⟨(equivYoneda S.1 S.2).hom⟩
  inv := ⟨(equivYoneda S.1 S.2).inv⟩
                   /-
                     S : CategoryTheory.Sheaf CategoryTheory.typesGrothendieckTopology (Type u)
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp { val := (CategoryTheory.equivYoneda  …
                   -/
  hom_inv_id := by ext1; apply (equivYoneda S.1 S.2).hom_inv_id
                         /-
                           🎉 no goals
                         -/
                   /-
                     S : CategoryTheory.Sheaf CategoryTheory.typesGrothendieckTopology (Type u)
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp { val := (CategoryTheory.equivYoneda  …
                   -/
  inv_hom_id := by ext1; apply (equivYoneda S.1 S.2).inv_hom_id
                         /-
                           🎉 no goals
                         -/


theorem eval_app (S₁ S₂ : Sheaf typesGrothendieckTopology (Type u)) (f : S₁ ⟶ S₂) (α : Type u)
    (s : S₁.1.obj (op α)) (x : α) :
    eval S₂.1 α (f.val.app (op α) s) x = f.val.app (op PUnit) (eval S₁.1 α s x) :=
  (congr_fun (f.val.naturality (↾fun _ : PUnit => x).op) s).symm


/-- `yoneda'` induces an equivalence of category between `Type u` and
`Sheaf typesGrothendieckTopology (Type u)`. -/
@[simps!]
noncomputable def typeEquiv : Type u ≌ Sheaf typesGrothendieckTopology (Type u) where
  functor := yoneda'
  inverse := sheafToPresheaf _ _ ⋙ (evaluation _ _).obj (op PUnit)
  unitIso := NatIso.ofComponents
      (fun _α => -- α ≅ PUnit ⟶ α
        { hom := fun x _ => x
          inv := fun f => f PUnit.unit
          hom_inv_id := funext fun _ => rfl
          inv_hom_id := funext fun _ => funext fun y => PUnit.casesOn y rfl })
      fun _ => rfl
  counitIso := Iso.symm <|
      NatIso.ofComponents (fun S => equivYoneda' S) (fun {S₁ S₂} f => by
        /-
          S₁ S₂ : CategoryTheory.Sheaf CategoryTheory.typesGrothendieckTopology (Type u)
          f : Quiver.Hom S₁ S₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Category …
        -/
        ext ⟨α⟩ s
        /-
          case h.w.h.op.h
          S₁ S₂ : CategoryTheory.Sheaf CategoryTheory.typesGrothendieckTopology (Type u)
          f : Quiver.Hom S₁ S₂
          α : Type u
          s : ((CategoryTheory.Functor.id (CategoryTheory.Sheaf CategoryTheory.typesGrot …
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Categor …
        -/
        dsimp at s ⊢
        /-
          case h.w.h.op.h
          S₁ S₂ : CategoryTheory.Sheaf CategoryTheory.typesGrothendieckTopology (Type u)
          f : Quiver.Hom S₁ S₂
          α : Type u
          s : S₁.val.obj { unop := α }
          ⊢ Eq ((CategoryTheory.evalEquiv S₂.val ⋯ α) (f.val.app { unop := α } s)) (Cate …
        -/
        ext x
        /-
          case h.w.h.op.h.h
          S₁ S₂ : CategoryTheory.Sheaf CategoryTheory.typesGrothendieckTopology (Type u)
          f : Quiver.Hom S₁ S₂
          α : Type u
          s : S₁.val.obj { unop := α }
          x : α
          ⊢ Eq ((CategoryTheory.evalEquiv S₂.val ⋯ α) (f.val.app { unop := α } s) x) (Ca …
        -/
        exact eval_app S₁ S₂ f α s x)
        /-
          🎉 no goals
        -/
  functor_unitIso_comp X := by
    /-
      X : Type u
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda'.map ((Categor …
    -/
    ext1
    /-
      case h
      X : Type u
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.yoneda'.map ((Categor …
    -/
    apply yonedaEquiv.injective
    /-
      case h.a
      X : Type u
      ⊢ Eq (CategoryTheory.yonedaEquiv (CategoryTheory.CategoryStruct.comp (Category …
    -/
    dsimp [yoneda', yonedaEquiv, evalEquiv]
    /-
      case h.a
      X : Type u
      ⊢ Eq (CategoryTheory.typesGlue (CategoryTheory.yoneda.obj X) ⋯ X (CategoryTheo …
    -/
    erw [typesGlue_eval]
    /-
      🎉 no goals
    -/


instance subcanonical_typesGrothendieckTopology : typesGrothendieckTopology.{u}.Subcanonical :=
  GrothendieckTopology.Subcanonical.of_isSheaf_yoneda_obj _ fun _ => Presieve.isSheaf_yoneda'


theorem typesGrothendieckTopology_eq_canonical :
    typesGrothendieckTopology.{u} = Sheaf.canonicalTopology (Type u) := by
  /-
    ⊢ Eq CategoryTheory.typesGrothendieckTopology (CategoryTheory.Sheaf.canonicalT …
  -/
  refine le_antisymm typesGrothendieckTopology.le_canonical (sInf_le ?_)
  /-
    ⊢ Membership.mem (Set.image CategoryTheory.Sheaf.finestTopologySingle (Set.ran …
  -/
  refine ⟨yoneda.obj (ULift Bool), ⟨_, rfl⟩, GrothendieckTopology.ext ?_⟩
  /-
    ⊢ Eq ⇑(CategoryTheory.Sheaf.finestTopologySingle (CategoryTheory.yoneda.obj (U …
  -/
  funext α
  /-
    case h
    α : Type u
    ⊢ Eq ((CategoryTheory.Sheaf.finestTopologySingle (CategoryTheory.yoneda.obj (U …
  -/
  ext S
  /-
    case h.h
    α : Type u
    S : CategoryTheory.Sieve α
    ⊢ Iff (Membership.mem ((CategoryTheory.Sheaf.finestTopologySingle (CategoryThe …
  -/
  refine ⟨fun hs x => ?_, fun hs β f => Presieve.isSheaf_yoneda' _ fun y => hs _⟩
  /-
    case h.h
    α : Type u
    S : CategoryTheory.Sieve α
    hs : Membership.mem ((CategoryTheory.Sheaf.finestTopologySingle (CategoryTheor …
    x : α
    ⊢ S.arrows fun x_1 => x
  -/
  by_contra hsx
  have : (fun _ => ULift.up true) = fun _ => ULift.up false :=
    (hs PUnit fun _ => x).isSeparatedFor.ext
      fun β f hf => funext fun y => hsx.elim <| S.2 hf fun _ => y
  /-
    case h.h
    α : Type u
    S : CategoryTheory.Sieve α
    hs : Membership.mem ((CategoryTheory.Sheaf.finestTopologySingle (CategoryTheor …
    x : α
    hsx : Not (S.arrows fun x_1 => x)
    this : Eq (fun x => { down := Bool.true }) fun x => { down := Bool.false }
    ⊢ False
  -/
  simp [funext_iff] at this
  /-
    🎉 no goals
  -/


