/-- traverse on the first functor argument -/
abbrev tfst {α α'} (f : α → F α') : t α β → F (t α' β) :=
  bitraverse f pure


/-- traverse on the second functor argument -/
abbrev tsnd {α α'} (f : α → F α') : t β α → F (t β α') :=
  bitraverse pure f


@[higher_order tfst_id]
theorem id_tfst : ∀ {α β} (x : t α β), tfst (F := Id) pure x = pure x :=
  id_bitraverse


@[higher_order tsnd_id]
theorem id_tsnd : ∀ {α β} (x : t α β), tsnd (F := Id) pure x = pure x :=
  id_bitraverse


@[higher_order tfst_comp_tfst]
theorem comp_tfst {α₀ α₁ α₂ β} (f : α₀ → F α₁) (f' : α₁ → G α₂) (x : t α₀ β) :
    Comp.mk (tfst f' <$> tfst f x) = tfst (Comp.mk ∘ map f' ∘ f) x := by
  /-
    t : Type u → Type u → Type u
    inst✝⁵ : Bitraversable t
    F G : Type u → Type u
    inst✝⁴ : Applicative F
    inst✝³ : Applicative G
    inst✝² : LawfulBitraversable t
    inst✝¹ : LawfulApplicative F
    inst✝ : LawfulApplicative G
    α₀ α₁ α₂ β : Type u
    f : α₀ → F α₁
    f' : α₁ → G α₂
    x : t α₀ β
    ⊢ Eq (Functor.Comp.mk (Functor.map (Bitraversable.tfst f') (Bitraversable.tfst …
  -/
  rw [← comp_bitraverse]
  /-
    t : Type u → Type u → Type u
    inst✝⁵ : Bitraversable t
    F G : Type u → Type u
    inst✝⁴ : Applicative F
    inst✝³ : Applicative G
    inst✝² : LawfulBitraversable t
    inst✝¹ : LawfulApplicative F
    inst✝ : LawfulApplicative G
    α₀ α₁ α₂ β : Type u
    f : α₀ → F α₁
    f' : α₁ → G α₂
    x : t α₀ β
    ⊢ Eq (Bitraversable.bitraverse (Function.comp Functor.Comp.mk (Function.comp ( …
  -/
  simp only [Function.comp_def, tfst, map_pure, Pure.pure]
  /-
    🎉 no goals
  -/


@[higher_order tfst_comp_tsnd]
theorem tfst_tsnd {α₀ α₁ β₀ β₁} (f : α₀ → F α₁) (f' : β₀ → G β₁) (x : t α₀ β₀) :
    Comp.mk (tfst f <$> tsnd f' x)
      = bitraverse (Comp.mk ∘ pure ∘ f) (Comp.mk ∘ map pure ∘ f') x := by
  /-
    t : Type u → Type u → Type u
    inst✝⁵ : Bitraversable t
    F G : Type u → Type u
    inst✝⁴ : Applicative F
    inst✝³ : Applicative G
    inst✝² : LawfulBitraversable t
    inst✝¹ : LawfulApplicative F
    inst✝ : LawfulApplicative G
    α₀ α₁ β₀ β₁ : Type u
    f : α₀ → F α₁
    f' : β₀ → G β₁
    x : t α₀ β₀
    ⊢ Eq (Functor.Comp.mk (Functor.map (Bitraversable.tfst f) (Bitraversable.tsnd  …
  -/
  rw [← comp_bitraverse]
  /-
    t : Type u → Type u → Type u
    inst✝⁵ : Bitraversable t
    F G : Type u → Type u
    inst✝⁴ : Applicative F
    inst✝³ : Applicative G
    inst✝² : LawfulBitraversable t
    inst✝¹ : LawfulApplicative F
    inst✝ : LawfulApplicative G
    α₀ α₁ β₀ β₁ : Type u
    f : α₀ → F α₁
    f' : β₀ → G β₁
    x : t α₀ β₀
    ⊢ Eq (Bitraversable.bitraverse (Function.comp Functor.Comp.mk (Function.comp ( …
  -/
  simp only [Function.comp_def, map_pure]
  /-
    🎉 no goals
  -/


@[higher_order tsnd_comp_tfst]
theorem tsnd_tfst {α₀ α₁ β₀ β₁} (f : α₀ → F α₁) (f' : β₀ → G β₁) (x : t α₀ β₀) :
    Comp.mk (tsnd f' <$> tfst f x)
      = bitraverse (Comp.mk ∘ map pure ∘ f) (Comp.mk ∘ pure ∘ f') x := by
  /-
    t : Type u → Type u → Type u
    inst✝⁵ : Bitraversable t
    F G : Type u → Type u
    inst✝⁴ : Applicative F
    inst✝³ : Applicative G
    inst✝² : LawfulBitraversable t
    inst✝¹ : LawfulApplicative F
    inst✝ : LawfulApplicative G
    α₀ α₁ β₀ β₁ : Type u
    f : α₀ → F α₁
    f' : β₀ → G β₁
    x : t α₀ β₀
    ⊢ Eq (Functor.Comp.mk (Functor.map (Bitraversable.tsnd f') (Bitraversable.tfst …
  -/
  rw [← comp_bitraverse]
  /-
    t : Type u → Type u → Type u
    inst✝⁵ : Bitraversable t
    F G : Type u → Type u
    inst✝⁴ : Applicative F
    inst✝³ : Applicative G
    inst✝² : LawfulBitraversable t
    inst✝¹ : LawfulApplicative F
    inst✝ : LawfulApplicative G
    α₀ α₁ β₀ β₁ : Type u
    f : α₀ → F α₁
    f' : β₀ → G β₁
    x : t α₀ β₀
    ⊢ Eq (Bitraversable.bitraverse (Function.comp Functor.Comp.mk (Function.comp ( …
  -/
  simp only [Function.comp_def, map_pure]
  /-
    🎉 no goals
  -/


@[higher_order tsnd_comp_tsnd]
theorem comp_tsnd {α β₀ β₁ β₂} (g : β₀ → F β₁) (g' : β₁ → G β₂) (x : t α β₀) :
    Comp.mk (tsnd g' <$> tsnd g x) = tsnd (Comp.mk ∘ map g' ∘ g) x := by
  /-
    t : Type u → Type u → Type u
    inst✝⁵ : Bitraversable t
    F G : Type u → Type u
    inst✝⁴ : Applicative F
    inst✝³ : Applicative G
    inst✝² : LawfulBitraversable t
    inst✝¹ : LawfulApplicative F
    inst✝ : LawfulApplicative G
    α β₀ β₁ β₂ : Type u
    g : β₀ → F β₁
    g' : β₁ → G β₂
    x : t α β₀
    ⊢ Eq (Functor.Comp.mk (Functor.map (Bitraversable.tsnd g') (Bitraversable.tsnd …
  -/
  rw [← comp_bitraverse]
  /-
    t : Type u → Type u → Type u
    inst✝⁵ : Bitraversable t
    F G : Type u → Type u
    inst✝⁴ : Applicative F
    inst✝³ : Applicative G
    inst✝² : LawfulBitraversable t
    inst✝¹ : LawfulApplicative F
    inst✝ : LawfulApplicative G
    α β₀ β₁ β₂ : Type u
    g : β₀ → F β₁
    g' : β₁ → G β₂
    x : t α β₀
    ⊢ Eq (Bitraversable.bitraverse (Function.comp Functor.Comp.mk (Function.comp ( …
  -/
  simp only [Function.comp_def, map_pure]
  /-
    t : Type u → Type u → Type u
    inst✝⁵ : Bitraversable t
    F G : Type u → Type u
    inst✝⁴ : Applicative F
    inst✝³ : Applicative G
    inst✝² : LawfulBitraversable t
    inst✝¹ : LawfulApplicative F
    inst✝ : LawfulApplicative G
    α β₀ β₁ β₂ : Type u
    g : β₀ → F β₁
    g' : β₁ → G β₂
    x : t α β₀
    ⊢ Eq (Bitraversable.bitraverse (fun x => Functor.Comp.mk (Pure.pure (Pure.pure …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[higher_order]
theorem tfst_eq_fst_id {α α' β} (f : α → α') (x : t α β) :
    tfst (F := Id) (pure ∘ f) x = pure (fst f x) := by
  /-
    t : Type u → Type u → Type u
    inst✝¹ : Bitraversable t
    inst✝ : LawfulBitraversable t
    α α' β : Type u
    f : α → α'
    x : t α β
    ⊢ Eq (Bitraversable.tfst (Function.comp Pure.pure f) x) (Pure.pure (Bifunctor. …
  -/
  apply bitraverse_eq_bimap_id
  /-
    🎉 no goals
  -/


@[higher_order]
theorem tsnd_eq_snd_id {α β β'} (f : β → β') (x : t α β) :
    tsnd (F := Id) (pure ∘ f) x = pure (snd f x) := by
  /-
    t : Type u → Type u → Type u
    inst✝¹ : Bitraversable t
    inst✝ : LawfulBitraversable t
    α β β' : Type u
    f : β → β'
    x : t α β
    ⊢ Eq (Bitraversable.tsnd (Function.comp Pure.pure f) x) (Pure.pure (Bifunctor. …
  -/
  apply bitraverse_eq_bimap_id
  /-
    🎉 no goals
  -/


