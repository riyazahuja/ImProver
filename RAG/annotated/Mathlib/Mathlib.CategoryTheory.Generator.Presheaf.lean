/-- Given `X : C` and `M : A`, this is the presheaf `Cᵒᵖ ⥤ A` which sends
`Y : Cᵒᵖ` to the coproduct of copies of `M` indexed by `Y.unop ⟶ X`. -/
@[simps]
noncomputable def freeYoneda (X : C) (M : A) : Cᵒᵖ ⥤ A where
  obj Y := ∐ (fun (i : (yoneda.obj X).obj Y) ↦ M)
  map f := Sigma.map' ((yoneda.obj X).map f) (fun _ ↦ 𝟙 M)


/-- The bijection `(Presheaf.freeYoneda X M ⟶ F) ≃ (M ⟶ F.obj (op X))`. -/
noncomputable def freeYonedaHomEquiv {X : C} {M : A} {F : Cᵒᵖ ⥤ A} :
    (freeYoneda X M ⟶ F) ≃ (M ⟶ F.obj (op X)) where
  toFun f := Sigma.ι (fun (i : (yoneda.obj X).obj _) ↦ M) (𝟙 _) ≫ f.app (op X)
  invFun g :=
    { app Y := Sigma.desc (fun φ ↦ g ≫ F.map φ.op)
                                                /-
                                                  C : Type u
                                                  inst✝² : CategoryTheory.Category.{v, u} C
                                                  A : Type u'
                                                  inst✝¹ : CategoryTheory.Category.{v', u'} A
                                                  inst✝ : CategoryTheory.Limits.HasCoproducts A
                                                  X : C
                                                  M : A
                                                  F : CategoryTheory.Functor (Opposite C) A
                                                  g : Quiver.Hom M (F.obj { unop := X })
                                                  x✝² x✝¹ : Opposite C
                                                  x✝ : Quiver.Hom x✝² x✝¹
                                                  ⊢ ∀ (b : (CategoryTheory.yoneda.obj X).obj x✝²), Eq (CategoryTheory.CategorySt …
                                                -/
      naturality _ _ _ := Sigma.hom_ext _ _ (by simp)}
                                                /-
                                                  🎉 no goals
                                                -/
  left_inv f := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.Limits.HasCoproducts A
      X : C
      M : A
      F : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom (CategoryTheory.Presheaf.freeYoneda X M) F
      ⊢ Eq ((fun g => { app := fun Y => CategoryTheory.Limits.Sigma.desc fun φ => Ca …
    -/
    ext Y
    /-
      case w.h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.Limits.HasCoproducts A
      X : C
      M : A
      F : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom (CategoryTheory.Presheaf.freeYoneda X M) F
      Y : Opposite C
      ⊢ Eq (((fun g => { app := fun Y => CategoryTheory.Limits.Sigma.desc fun φ => C …
    -/
    refine Sigma.hom_ext _ _ (fun φ ↦ ?_)
    /-
      case w.h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      A : Type u'
      inst✝¹ : CategoryTheory.Category.{v', u'} A
      inst✝ : CategoryTheory.Limits.HasCoproducts A
      X : C
      M : A
      F : CategoryTheory.Functor (Opposite C) A
      f : Quiver.Hom (CategoryTheory.Presheaf.freeYoneda X M) F
      Y : Opposite C
      φ : (CategoryTheory.yoneda.obj X).obj Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι (fun i …
    -/
    simpa using (Sigma.ι _ (𝟙 _) ≫= f.naturality φ.op).symm
    /-
      🎉 no goals
    -/
                    /-
                      C : Type u
                      inst✝² : CategoryTheory.Category.{v, u} C
                      A : Type u'
                      inst✝¹ : CategoryTheory.Category.{v', u'} A
                      inst✝ : CategoryTheory.Limits.HasCoproducts A
                      X : C
                      M : A
                      F : CategoryTheory.Functor (Opposite C) A
                      g : Quiver.Hom M (F.obj { unop := X })
                      ⊢ Eq ((fun f => CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigm …
                    -/
  right_inv g := by simp
                    /-
                      🎉 no goals
                    -/


@[reassoc]
lemma freeYonedaHomEquiv_comp {X : C} {M : A} {F G : Cᵒᵖ ⥤ A}
    (α : freeYoneda X M ⟶ F) (f : F ⟶ G) :
    freeYonedaHomEquiv (α ≫ f) = freeYonedaHomEquiv α ≫ f.app (op X) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.Limits.HasCoproducts A
    X : C
    M : A
    F G : CategoryTheory.Functor (Opposite C) A
    α : Quiver.Hom (CategoryTheory.Presheaf.freeYoneda X M) F
    f : Quiver.Hom F G
    ⊢ Eq (CategoryTheory.Presheaf.freeYonedaHomEquiv (CategoryTheory.CategoryStruc …
  -/
  simp [freeYonedaHomEquiv]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma freeYonedaHomEquiv_symm_comp {X : C} {M : A} {F G : Cᵒᵖ ⥤ A} (α : M ⟶ F.obj (op X))
    (f : F ⟶ G) :
    freeYonedaHomEquiv.symm α ≫ f = freeYonedaHomEquiv.symm (α ≫ f.app (op X)) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.Limits.HasCoproducts A
    X : C
    M : A
    F G : CategoryTheory.Functor (Opposite C) A
    α : Quiver.Hom M (F.obj { unop := X })
    f : Quiver.Hom F G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Presheaf.freeYonedaHo …
  -/
  obtain ⟨β, rfl⟩ := freeYonedaHomEquiv.surjective α
  /-
    case intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.Limits.HasCoproducts A
    X : C
    M : A
    F G : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom F G
    β : Quiver.Hom (CategoryTheory.Presheaf.freeYoneda X M) F
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Presheaf.freeYonedaHo …
  -/
  apply freeYonedaHomEquiv.injective
  /-
    case intro.a
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.Limits.HasCoproducts A
    X : C
    M : A
    F G : CategoryTheory.Functor (Opposite C) A
    f : Quiver.Hom F G
    β : Quiver.Hom (CategoryTheory.Presheaf.freeYoneda X M) F
    ⊢ Eq (CategoryTheory.Presheaf.freeYonedaHomEquiv (CategoryTheory.CategoryStruc …
  -/
  simp only [Equiv.symm_apply_apply, freeYonedaHomEquiv_comp, Equiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


lemma isSeparating {ι : Type w} {S : ι → A} (hS : IsSeparating (Set.range S)) :
    IsSeparating (Set.range (fun (⟨X, i⟩ : C × ι) ↦ freeYoneda X (S i))) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.Limits.HasCoproducts A
    ι : Type w
    S : ι → A
    hS : CategoryTheory.IsSeparating (Set.range S)
    ⊢ CategoryTheory.IsSeparating (Set.range fun x => CategoryTheory.Presheaf.isSe …
  -/
  intro F G f g h
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.Limits.HasCoproducts A
    ι : Type w
    S : ι → A
    hS : CategoryTheory.IsSeparating (Set.range S)
    F G : CategoryTheory.Functor (Opposite C) A
    f g : Quiver.Hom F G
    h : ∀ (G_1 : CategoryTheory.Functor (Opposite C) A), Membership.mem (Set.range …
    ⊢ Eq f g
  -/
  ext ⟨X⟩
  /-
    case w.h.op
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.Limits.HasCoproducts A
    ι : Type w
    S : ι → A
    hS : CategoryTheory.IsSeparating (Set.range S)
    F G : CategoryTheory.Functor (Opposite C) A
    f g : Quiver.Hom F G
    h : ∀ (G_1 : CategoryTheory.Functor (Opposite C) A), Membership.mem (Set.range …
    X : C
    ⊢ Eq (f.app { unop := X }) (g.app { unop := X })
  -/
  refine hS _ _ ?_
  /-
    case w.h.op
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.Limits.HasCoproducts A
    ι : Type w
    S : ι → A
    hS : CategoryTheory.IsSeparating (Set.range S)
    F G : CategoryTheory.Functor (Opposite C) A
    f g : Quiver.Hom F G
    h : ∀ (G_1 : CategoryTheory.Functor (Opposite C) A), Membership.mem (Set.range …
    X : C
    ⊢ ∀ (G_1 : A), Membership.mem (Set.range S) G_1 → ∀ (h : Quiver.Hom G_1 (F.obj …
  -/
  rintro _ ⟨i, rfl⟩ α
  /-
    case w.h.op.intro
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    A : Type u'
    inst✝¹ : CategoryTheory.Category.{v', u'} A
    inst✝ : CategoryTheory.Limits.HasCoproducts A
    ι : Type w
    S : ι → A
    hS : CategoryTheory.IsSeparating (Set.range S)
    F G : CategoryTheory.Functor (Opposite C) A
    f g : Quiver.Hom F G
    h : ∀ (G_1 : CategoryTheory.Functor (Opposite C) A), Membership.mem (Set.range …
    X : C
    i : ι
    α : Quiver.Hom (S i) (F.obj { unop := X })
    ⊢ Eq (CategoryTheory.CategoryStruct.comp α (f.app { unop := X })) (CategoryThe …
  -/
  apply freeYonedaHomEquiv.symm.injective
  simpa only [freeYonedaHomEquiv_symm_comp] using
    h _ ⟨⟨X, i⟩, rfl⟩ (freeYonedaHomEquiv.symm α)


lemma isSeparator {ι : Type w} {S : ι → A} (hS : IsSeparating (Set.range S))
    [HasCoproduct (fun (⟨X, i⟩ : C × ι) ↦ freeYoneda X (S i))]
    [HasZeroMorphisms A] :
    IsSeparator (∐ (fun (⟨X, i⟩ : C × ι) ↦ freeYoneda X (S i))) :=
  (isSeparating C hS).isSeparator_coproduct


variable (A) in
instance hasSeparator [HasSeparator A] [HasZeroMorphisms A] [HasCoproducts.{u} A] :
    HasSeparator (Cᵒᵖ ⥤ A) where
  hasSeparator := ⟨_, isSeparator C (S := fun (_ : Unit) ↦ separator A)
          /-
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            A : Type u'
            inst✝⁴ : CategoryTheory.Category.{v', u'} A
            inst✝³ : CategoryTheory.Limits.HasCoproducts A
            inst✝² : CategoryTheory.HasSeparator A
            inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms A
            inst✝ : CategoryTheory.Limits.HasCoproducts A
            ⊢ CategoryTheory.IsSeparating (Set.range fun x => CategoryTheory.separator A)
          -/
      (by simpa using isSeparator_separator A)⟩
          /-
            🎉 no goals
          -/


