variable (I) in
/--
If `F` is a presheaf which satisfies the sheaf condition with respect to the empty presieve on any
object, then `F` takes that object to the terminal object.
-/
noncomputable
def isTerminal_of_isSheafFor_empty_presieve : IsTerminal (F.obj (op I)) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    I : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
    ⊢ CategoryTheory.Limits.IsTerminal (F.obj { unop := I })
  -/
  refine @IsTerminal.ofUnique _ _ _ fun Y ↦ ?_
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    I : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
    Y : Type w
    ⊢ Unique (Quiver.Hom Y (F.obj { unop := I }))
  -/
  choose t h using hF (by tauto) (by tauto)
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    I : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
    Y : Type w
    t : F.obj { unop := I }
    h : And ((fun t => CategoryTheory.Presieve.FamilyOfElements.IsAmalgamation (fu …
    ⊢ Unique (Quiver.Hom Y (F.obj { unop := I }))
  -/
  exact ⟨⟨fun _ ↦ t⟩, fun a ↦ by ext; exact h.2 _ (by tauto)⟩
  /-
    🎉 no goals
  -/


include hF in
/--
If `F` is a presheaf which satisfies the sheaf condition with respect to the empty presieve on the
initial object, then `F` preserves terminal objects.
-/
lemma preservesTerminal_of_isSheaf_for_empty (hI : IsInitial I) :
    PreservesLimit (Functor.empty.{0} Cᵒᵖ) F :=
  have := hI.hasInitial
  (preservesTerminal_of_iso F
    ((F.mapIso (terminalIsoIsTerminal (terminalOpOfInitial initialIsInitial)) ≪≫
    (F.mapIso (initialIsoIsInitial hI).symm.op) ≪≫
    (terminalIsoIsTerminal (isTerminal_of_isSheafFor_empty_presieve I F hF)).symm)))


theorem piComparison_fac :
    have : HasCoproduct X := ⟨⟨c, hc⟩⟩
    piComparison F (fun x ↦ op (X x)) = F.map (opCoproductIsoProduct' hc (productIsProduct _)).inv ≫
    Equalizer.Presieve.Arrows.forkMap F X c.inj := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    ⊢ letFun ⋯ fun this => Eq (CategoryTheory.Limits.piComparison F fun x => { uno …
  -/
  have : HasCoproduct X := ⟨⟨c, hc⟩⟩
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    this : CategoryTheory.Limits.HasCoproduct X
    ⊢ letFun ⋯ fun this => Eq (CategoryTheory.Limits.piComparison F fun x => { uno …
  -/
  dsimp only [Equalizer.Presieve.Arrows.forkMap]
  have h : Pi.lift (fun i ↦ F.map (c.inj i).op) =
      F.map (Pi.lift (fun i ↦ (c.inj i).op)) ≫ piComparison F _ := by simp
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    this : CategoryTheory.Limits.HasCoproduct X
    h : Eq (CategoryTheory.Limits.Pi.lift fun i => F.map (c.inj i).op) (CategoryTh …
    ⊢ Eq (CategoryTheory.Limits.piComparison F fun x => { unop := X x }) (Category …
  -/
  rw [h, ← Category.assoc, ← Functor.map_comp]
  have hh : Pi.lift (fun i ↦ (c.inj i).op) = (productIsProduct (op <| X ·)).lift c.op := by
    simp [Pi.lift, productIsProduct]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    this : CategoryTheory.Limits.HasCoproduct X
    h : Eq (CategoryTheory.Limits.Pi.lift fun i => F.map (c.inj i).op) (CategoryTh …
    hh : Eq (CategoryTheory.Limits.Pi.lift fun i => (c.inj i).op) ((CategoryTheory …
    ⊢ Eq (CategoryTheory.Limits.piComparison F fun x => { unop := X x }) (Category …
  -/
  rw [hh, ← desc_op_comp_opCoproductIsoProduct'_hom hc]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    this : CategoryTheory.Limits.HasCoproduct X
    h : Eq (CategoryTheory.Limits.Pi.lift fun i => F.map (c.inj i).op) (CategoryTh …
    hh : Eq (CategoryTheory.Limits.Pi.lift fun i => (c.inj i).op) ((CategoryTheory …
    ⊢ Eq (CategoryTheory.Limits.piComparison F fun x => { unop := X x }) (Category …
  -/
  simp
  /-
    🎉 no goals
  -/


include hc in
/--
If `F` preserves a particular product, then it `IsSheafFor` the corresponding presieve of arrows.
-/
theorem isSheafFor_of_preservesProduct [PreservesLimit (Discrete.functor (fun x ↦ op (X x))) F] :
    (ofArrows X c.inj).IsSheafFor F := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝¹ : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    ⊢ CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows X c.i …
  -/
  rw [Equalizer.Presieve.Arrows.sheaf_condition, Limits.Types.type_equalizer_iff_unique]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝¹ : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    ⊢ ∀ (y : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X), Eq (CategoryT …
  -/
  have : HasCoproduct X := ⟨⟨c, hc⟩⟩
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝¹ : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    this : CategoryTheory.Limits.HasCoproduct X
    ⊢ ∀ (y : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X), Eq (CategoryT …
  -/
  have hi : IsIso (piComparison F (fun x ↦ op (X x))) := inferInstance
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝¹ : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    this : CategoryTheory.Limits.HasCoproduct X
    hi : CategoryTheory.IsIso (CategoryTheory.Limits.piComparison F fun x => { uno …
    ⊢ ∀ (y : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X), Eq (CategoryT …
  -/
  rw [piComparison_fac (hc := hc), isIso_iff_bijective, Function.bijective_iff_existsUnique] at hi
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝¹ : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    this : CategoryTheory.Limits.HasCoproduct X
    hi : ∀ (b : CategoryTheory.Limits.piObj fun b => F.obj { unop := X b }), Exist …
    ⊢ ∀ (y : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X), Eq (CategoryT …
  -/
  intro b _
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝¹ : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    this : CategoryTheory.Limits.HasCoproduct X
    hi : ∀ (b : CategoryTheory.Limits.piObj fun b => F.obj { unop := X b }), Exist …
    b : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X
    a✝ : Eq (CategoryTheory.Equalizer.Presieve.Arrows.firstMap F X c.inj b) (Categ …
    ⊢ ExistsUnique fun x => Eq (CategoryTheory.Equalizer.Presieve.Arrows.forkMap F …
  -/
  obtain ⟨t, ht₁, ht₂⟩ := hi b
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝¹ : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    this : CategoryTheory.Limits.HasCoproduct X
    hi : ∀ (b : CategoryTheory.Limits.piObj fun b => F.obj { unop := X b }), Exist …
    b : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X
    a✝ : Eq (CategoryTheory.Equalizer.Presieve.Arrows.firstMap F X c.inj b) (Categ …
    t : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })
    ht₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.opC …
    ht₂ : ∀ (y : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })), (f …
    ⊢ ExistsUnique fun x => Eq (CategoryTheory.Equalizer.Presieve.Arrows.forkMap F …
  -/
  refine ⟨F.map ((opCoproductIsoProduct' hc (productIsProduct _)).inv) t, ht₁, fun y hy ↦ ?_⟩
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝¹ : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    this : CategoryTheory.Limits.HasCoproduct X
    hi : ∀ (b : CategoryTheory.Limits.piObj fun b => F.obj { unop := X b }), Exist …
    b : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X
    a✝ : Eq (CategoryTheory.Equalizer.Presieve.Arrows.firstMap F X c.inj b) (Categ …
    t : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })
    ht₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.opC …
    ht₂ : ∀ (y : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })), (f …
    y : F.obj { unop := c.pt }
    hy : (fun x => Eq (CategoryTheory.Equalizer.Presieve.Arrows.forkMap F X c.inj  …
    ⊢ Eq y (F.map (CategoryTheory.Limits.opCoproductIsoProduct' hc (CategoryTheory …
  -/
  apply_fun F.map ((opCoproductIsoProduct' hc (productIsProduct _)).hom) using injective_of_mono _
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝¹ : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    this : CategoryTheory.Limits.HasCoproduct X
    hi : ∀ (b : CategoryTheory.Limits.piObj fun b => F.obj { unop := X b }), Exist …
    b : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X
    a✝ : Eq (CategoryTheory.Equalizer.Presieve.Arrows.firstMap F X c.inj b) (Categ …
    t : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })
    ht₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.opC …
    ht₂ : ∀ (y : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })), (f …
    y : F.obj { unop := c.pt }
    hy : (fun x => Eq (CategoryTheory.Equalizer.Presieve.Arrows.forkMap F X c.inj  …
    ⊢ Eq (F.map (CategoryTheory.Limits.opCoproductIsoProduct' hc (CategoryTheory.L …
  -/
  simp only [← FunctorToTypes.map_comp_apply, Iso.op, Category.assoc]
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝¹ : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    this : CategoryTheory.Limits.HasCoproduct X
    hi : ∀ (b : CategoryTheory.Limits.piObj fun b => F.obj { unop := X b }), Exist …
    b : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X
    a✝ : Eq (CategoryTheory.Equalizer.Presieve.Arrows.firstMap F X c.inj b) (Categ …
    t : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })
    ht₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.opC …
    ht₂ : ∀ (y : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })), (f …
    y : F.obj { unop := c.pt }
    hy : (fun x => Eq (CategoryTheory.Equalizer.Presieve.Arrows.forkMap F X c.inj  …
    ⊢ Eq (F.map (CategoryTheory.Limits.opCoproductIsoProduct' hc (CategoryTheory.L …
  -/
  rw [ht₂ (F.map ((opCoproductIsoProduct' hc (productIsProduct _)).hom) y) (by simp [← hy])]
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝¹ : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    this : CategoryTheory.Limits.HasCoproduct X
    hi : ∀ (b : CategoryTheory.Limits.piObj fun b => F.obj { unop := X b }), Exist …
    b : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X
    a✝ : Eq (CategoryTheory.Equalizer.Presieve.Arrows.firstMap F X c.inj b) (Categ …
    t : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })
    ht₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.opC …
    ht₂ : ∀ (y : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })), (f …
    y : F.obj { unop := c.pt }
    hy : (fun x => Eq (CategoryTheory.Equalizer.Presieve.Arrows.forkMap F X c.inj  …
    ⊢ Eq t (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.opCop …
  -/
  change (𝟙 (F.obj (∏ᶜ fun x ↦ op (X x)))) t = _
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝¹ : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    this : CategoryTheory.Limits.HasCoproduct X
    hi : ∀ (b : CategoryTheory.Limits.piObj fun b => F.obj { unop := X b }), Exist …
    b : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X
    a✝ : Eq (CategoryTheory.Equalizer.Presieve.Arrows.firstMap F X c.inj b) (Categ …
    t : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })
    ht₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.opC …
    ht₂ : ∀ (y : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })), (f …
    y : F.obj { unop := c.pt }
    hy : (fun x => Eq (CategoryTheory.Equalizer.Presieve.Arrows.forkMap F X c.inj  …
    ⊢ Eq (CategoryTheory.CategoryStruct.id (F.obj (CategoryTheory.Limits.piObj fun …
  -/
  rw [← Functor.map_id]
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝¹ : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    this : CategoryTheory.Limits.HasCoproduct X
    hi : ∀ (b : CategoryTheory.Limits.piObj fun b => F.obj { unop := X b }), Exist …
    b : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X
    a✝ : Eq (CategoryTheory.Equalizer.Presieve.Arrows.firstMap F X c.inj b) (Categ …
    t : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })
    ht₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.opC …
    ht₂ : ∀ (y : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })), (f …
    y : F.obj { unop := c.pt }
    hy : (fun x => Eq (CategoryTheory.Equalizer.Presieve.Arrows.forkMap F X c.inj  …
    ⊢ Eq (F.map (CategoryTheory.CategoryStruct.id (CategoryTheory.Limits.piObj fun …
  -/
  refine congrFun ?_ t
  /-
    case intro.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝¹ : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    this : CategoryTheory.Limits.HasCoproduct X
    hi : ∀ (b : CategoryTheory.Limits.piObj fun b => F.obj { unop := X b }), Exist …
    b : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X
    a✝ : Eq (CategoryTheory.Equalizer.Presieve.Arrows.firstMap F X c.inj b) (Categ …
    t : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })
    ht₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.opC …
    ht₂ : ∀ (y : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })), (f …
    y : F.obj { unop := c.pt }
    hy : (fun x => Eq (CategoryTheory.Equalizer.Presieve.Arrows.forkMap F X c.inj  …
    ⊢ Eq (F.map (CategoryTheory.CategoryStruct.id (CategoryTheory.Limits.piObj fun …
  -/
  congr
  /-
    case intro.intro.e_a
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝¹ : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor  …
    this : CategoryTheory.Limits.HasCoproduct X
    hi : ∀ (b : CategoryTheory.Limits.piObj fun b => F.obj { unop := X b }), Exist …
    b : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X
    a✝ : Eq (CategoryTheory.Equalizer.Presieve.Arrows.firstMap F X c.inj b) (Categ …
    t : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })
    ht₁ : Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.Limits.opC …
    ht₂ : ∀ (y : F.obj (CategoryTheory.Limits.piObj fun x => { unop := X x })), (f …
    y : F.obj { unop := c.pt }
    hy : (fun x => Eq (CategoryTheory.Equalizer.Presieve.Arrows.forkMap F X c.inj  …
    ⊢ Eq (CategoryTheory.CategoryStruct.id (CategoryTheory.Limits.piObj fun x => { …
  -/
  simp [Iso.eq_inv_comp, ← Category.assoc, ← op_comp, eq_comm, ← Iso.eq_comp_inv]
  /-
    🎉 no goals
  -/


include hd hF hI in
/--
The two parallel maps in the equalizer diagram for the sheaf condition corresponding to the
inclusion maps in a disjoint coproduct are equal.
-/
theorem firstMap_eq_secondMap :
    Equalizer.Presieve.Arrows.firstMap F X c.inj =
    Equalizer.Presieve.Arrows.secondMap F X c.inj := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    I : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
    hI : CategoryTheory.Limits.IsInitial I
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    inst✝² : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : ∀ (i : α), CategoryTheory.Mono (c.inj i)
    hd : Pairwise fun i j => CategoryTheory.IsPullback (CategoryTheory.Limits.init …
    ⊢ Eq (CategoryTheory.Equalizer.Presieve.Arrows.firstMap F X c.inj) (CategoryTh …
  -/
  ext a ⟨i, j⟩
  simp only [Equalizer.Presieve.Arrows.firstMap, Types.pi_lift_π_apply, types_comp_apply,
    Equalizer.Presieve.Arrows.secondMap]
  /-
    case h.h.mk
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    I : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
    hI : CategoryTheory.Limits.IsInitial I
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    inst✝² : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : ∀ (i : α), CategoryTheory.Mono (c.inj i)
    hd : Pairwise fun i j => CategoryTheory.IsPullback (CategoryTheory.Limits.init …
    a : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X
    i j : α
    ⊢ Eq (F.map (CategoryTheory.Limits.pullback.fst (c.inj i) (c.inj j)).op (Categ …
  -/
  by_cases hi : i = j
    /-
      case pos
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      I : C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
      hI : CategoryTheory.Limits.IsInitial I
      α : Type
      X : α → C
      c : CategoryTheory.Limits.Cofan X
      inst✝² : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
      inst✝¹ : CategoryTheory.Limits.HasInitial C
      inst✝ : ∀ (i : α), CategoryTheory.Mono (c.inj i)
      hd : Pairwise fun i j => CategoryTheory.IsPullback (CategoryTheory.Limits.init …
      a : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X
      i j : α
      hi : Eq i j
      ⊢ Eq (F.map (CategoryTheory.Limits.pullback.fst (c.inj i) (c.inj j)).op (Categ …
    -/
  · rw [hi, Mono.right_cancellation _ _ pullback.condition]
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      I : C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
      hI : CategoryTheory.Limits.IsInitial I
      α : Type
      X : α → C
      c : CategoryTheory.Limits.Cofan X
      inst✝² : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
      inst✝¹ : CategoryTheory.Limits.HasInitial C
      inst✝ : ∀ (i : α), CategoryTheory.Mono (c.inj i)
      hd : Pairwise fun i j => CategoryTheory.IsPullback (CategoryTheory.Limits.init …
      a : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X
      i j : α
      hi : Not (Eq i j)
      ⊢ Eq (F.map (CategoryTheory.Limits.pullback.fst (c.inj i) (c.inj j)).op (Categ …
    -/
  · have := preservesTerminal_of_isSheaf_for_empty F hF hI
    apply_fun (F.mapIso ((hd hi).isoPullback).op ≪≫ F.mapIso (terminalIsoIsTerminal
      (terminalOpOfInitial initialIsInitial)).symm ≪≫ (PreservesTerminal.iso F)).hom using
      injective_of_mono _
    /-
      case neg
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      I : C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
      hI : CategoryTheory.Limits.IsInitial I
      α : Type
      X : α → C
      c : CategoryTheory.Limits.Cofan X
      inst✝² : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
      inst✝¹ : CategoryTheory.Limits.HasInitial C
      inst✝ : ∀ (i : α), CategoryTheory.Mono (c.inj i)
      hd : Pairwise fun i j => CategoryTheory.IsPullback (CategoryTheory.Limits.init …
      a : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X
      i j : α
      hi : Not (Eq i j)
      this : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty (Opp …
      ⊢ Eq (((F.mapIso (CategoryTheory.IsPullback.isoPullback ⋯).op).trans ((F.mapIs …
    -/
    ext ⟨i⟩
    /-
      case neg.w.mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      I : C
      F : CategoryTheory.Functor (Opposite C) (Type w)
      hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
      hI : CategoryTheory.Limits.IsInitial I
      α : Type
      X : α → C
      c : CategoryTheory.Limits.Cofan X
      inst✝² : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
      inst✝¹ : CategoryTheory.Limits.HasInitial C
      inst✝ : ∀ (i : α), CategoryTheory.Mono (c.inj i)
      hd : Pairwise fun i j => CategoryTheory.IsPullback (CategoryTheory.Limits.init …
      a : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X
      i✝ j : α
      hi : Not (Eq i✝ j)
      this : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Functor.empty (Opp …
      i : PEmpty.{1}
      ⊢ Eq (CategoryTheory.Limits.limit.π (CategoryTheory.Functor.empty (Type w)) {  …
    -/
    exact i.elim
    /-
      🎉 no goals
    -/


include hc hd hF hI in
/--
If `F` is a presheaf which `IsSheafFor` a presieve of arrows and the empty presieve, then it
preserves the product corresponding to the presieve of arrows.
-/
lemma preservesProduct_of_isSheafFor
    (hF' : (ofArrows X c.inj).IsSheafFor F) :
    PreservesLimit (Discrete.functor (fun x ↦ op (X x))) F := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    I : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
    hI : CategoryTheory.Limits.IsInitial I
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝² : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : ∀ (i : α), CategoryTheory.Mono (c.inj i)
    hd : Pairwise fun i j => CategoryTheory.IsPullback (CategoryTheory.Limits.init …
    hF' : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows X …
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor fun x  …
  -/
  have : HasCoproduct X := ⟨⟨c, hc⟩⟩
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    I : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
    hI : CategoryTheory.Limits.IsInitial I
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝² : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : ∀ (i : α), CategoryTheory.Mono (c.inj i)
    hd : Pairwise fun i j => CategoryTheory.IsPullback (CategoryTheory.Limits.init …
    hF' : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows X …
    this : CategoryTheory.Limits.HasCoproduct X
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor fun x  …
  -/
  refine @PreservesProduct.of_iso_comparison _ _ _ _ F _ (fun x ↦ op (X x)) _ _ ?_
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    I : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
    hI : CategoryTheory.Limits.IsInitial I
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝² : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : ∀ (i : α), CategoryTheory.Mono (c.inj i)
    hd : Pairwise fun i j => CategoryTheory.IsPullback (CategoryTheory.Limits.init …
    hF' : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows X …
    this : CategoryTheory.Limits.HasCoproduct X
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.piComparison F fun x => { unop : …
  -/
  rw [piComparison_fac (hc := hc)]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    I : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
    hI : CategoryTheory.Limits.IsInitial I
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝² : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : ∀ (i : α), CategoryTheory.Mono (c.inj i)
    hd : Pairwise fun i j => CategoryTheory.IsPullback (CategoryTheory.Limits.init …
    hF' : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows X …
    this : CategoryTheory.Limits.HasCoproduct X
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (F.map (CategoryThe …
  -/
  refine IsIso.comp_isIso' inferInstance ?_
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    I : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
    hI : CategoryTheory.Limits.IsInitial I
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝² : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : ∀ (i : α), CategoryTheory.Mono (c.inj i)
    hd : Pairwise fun i j => CategoryTheory.IsPullback (CategoryTheory.Limits.init …
    hF' : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows X …
    this : CategoryTheory.Limits.HasCoproduct X
    ⊢ CategoryTheory.IsIso (CategoryTheory.Equalizer.Presieve.Arrows.forkMap F X c …
  -/
  rw [isIso_iff_bijective, Function.bijective_iff_existsUnique]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    I : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
    hI : CategoryTheory.Limits.IsInitial I
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝² : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : ∀ (i : α), CategoryTheory.Mono (c.inj i)
    hd : Pairwise fun i j => CategoryTheory.IsPullback (CategoryTheory.Limits.init …
    hF' : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows X …
    this : CategoryTheory.Limits.HasCoproduct X
    ⊢ ∀ (b : CategoryTheory.Limits.piObj fun b => F.obj ((fun x => { unop := X x } …
  -/
  rw [Equalizer.Presieve.Arrows.sheaf_condition, Limits.Types.type_equalizer_iff_unique] at hF'
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    I : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
    hI : CategoryTheory.Limits.IsInitial I
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝² : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : ∀ (i : α), CategoryTheory.Mono (c.inj i)
    hd : Pairwise fun i j => CategoryTheory.IsPullback (CategoryTheory.Limits.init …
    hF' : ∀ (y : CategoryTheory.Equalizer.Presieve.Arrows.FirstObj F X), Eq (Categ …
    this : CategoryTheory.Limits.HasCoproduct X
    ⊢ ∀ (b : CategoryTheory.Limits.piObj fun b => F.obj ((fun x => { unop := X x } …
  -/
  exact fun b ↦ hF' b (congr_fun (firstMap_eq_secondMap F hF hI c hd) b)
  /-
    🎉 no goals
  -/


include hc hd hF hI in
theorem isSheafFor_iff_preservesProduct : (ofArrows X c.inj).IsSheafFor F ↔
    Nonempty (PreservesLimit (Discrete.functor (fun x ↦ op (X x))) F) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    I : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
    hI : CategoryTheory.Limits.IsInitial I
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝² : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : ∀ (i : α), CategoryTheory.Mono (c.inj i)
    hd : Pairwise fun i j => CategoryTheory.IsPullback (CategoryTheory.Limits.init …
    ⊢ Iff (CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows  …
  -/
  refine ⟨fun hF' ↦ ⟨preservesProduct_of_isSheafFor _ hF hI c hc hd hF'⟩, fun hF' ↦ ?_⟩
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    I : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
    hI : CategoryTheory.Limits.IsInitial I
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝² : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : ∀ (i : α), CategoryTheory.Mono (c.inj i)
    hd : Pairwise fun i j => CategoryTheory.IsPullback (CategoryTheory.Limits.init …
    hF' : Nonempty (CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete. …
    ⊢ CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows X c.i …
  -/
  let _ := hF'.some
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    I : C
    F : CategoryTheory.Functor (Opposite C) (Type w)
    hF : CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows Em …
    hI : CategoryTheory.Limits.IsInitial I
    α : Type
    X : α → C
    c : CategoryTheory.Limits.Cofan X
    hc : CategoryTheory.Limits.IsColimit c
    inst✝² : (CategoryTheory.Presieve.ofArrows X c.inj).hasPullbacks
    inst✝¹ : CategoryTheory.Limits.HasInitial C
    inst✝ : ∀ (i : α), CategoryTheory.Mono (c.inj i)
    hd : Pairwise fun i j => CategoryTheory.IsPullback (CategoryTheory.Limits.init …
    hF' : Nonempty (CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete. …
    x✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Discrete.functor fun …
    ⊢ CategoryTheory.Presieve.IsSheafFor F (CategoryTheory.Presieve.ofArrows X c.i …
  -/
  exact isSheafFor_of_preservesProduct F c hc
  /-
    🎉 no goals
  -/


