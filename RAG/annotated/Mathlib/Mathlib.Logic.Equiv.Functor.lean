/-- Apply a functor to an `Equiv`. -/
def mapEquiv (h : α ≃ β) : f α ≃ f β where
  toFun := map h
  invFun := map h.symm
                   /-
                     α β : Type u
                     f : Type u → Type v
                     inst✝¹ : Functor f
                     inst✝ : LawfulFunctor f
                     h : Equiv α β
                     x : f α
                     ⊢ Eq (Functor.map (⇑h.symm) (Functor.map (⇑h) x)) x
                   -/
  left_inv x := by simp [map_map]
                   /-
                     🎉 no goals
                   -/
                    /-
                      α β : Type u
                      f : Type u → Type v
                      inst✝¹ : Functor f
                      inst✝ : LawfulFunctor f
                      h : Equiv α β
                      x : f β
                      ⊢ Eq (Functor.map (⇑h) (Functor.map (⇑h.symm) x)) x
                    -/
  right_inv x := by simp [map_map]
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem mapEquiv_apply (h : α ≃ β) (x : f α) : (mapEquiv f h : f α ≃ f β) x = map h x :=
  rfl


@[simp]
theorem mapEquiv_symm_apply (h : α ≃ β) (y : f β) :
    (mapEquiv f h : f α ≃ f β).symm y = map h.symm y :=
  rfl


@[simp]
theorem mapEquiv_refl : mapEquiv f (Equiv.refl α) = Equiv.refl (f α) := by
  /-
    α : Type u
    f : Type u → Type v
    inst✝¹ : Functor f
    inst✝ : LawfulFunctor f
    ⊢ Eq (Functor.mapEquiv f (Equiv.refl α)) (Equiv.refl (f α))
  -/
  ext x
  /-
    case H
    α : Type u
    f : Type u → Type v
    inst✝¹ : Functor f
    inst✝ : LawfulFunctor f
    x : f α
    ⊢ Eq ((Functor.mapEquiv f (Equiv.refl α)) x) ((Equiv.refl (f α)) x)
  -/
  simp only [mapEquiv_apply, refl_apply]
  /-
    case H
    α : Type u
    f : Type u → Type v
    inst✝¹ : Functor f
    inst✝ : LawfulFunctor f
    x : f α
    ⊢ Eq (Functor.map (⇑(Equiv.refl α)) x) x
  -/
  exact LawfulFunctor.id_map x
  /-
    🎉 no goals
  -/


/-- Apply a bifunctor to a pair of `Equiv`s. -/
def mapEquiv (h : α ≃ β) (h' : α' ≃ β') : F α α' ≃ F β β' where
  toFun := bimap h h'
  invFun := bimap h.symm h'.symm
                   /-
                     α β : Type u
                     α' β' : Type v
                     F : Type u → Type v → Type w
                     inst✝¹ : Bifunctor F
                     inst✝ : LawfulBifunctor F
                     h : Equiv α β
                     h' : Equiv α' β'
                     x : F α α'
                     ⊢ Eq (Bifunctor.bimap (⇑h.symm) (⇑h'.symm) (Bifunctor.bimap (⇑h) (⇑h') x)) x
                   -/
  left_inv x := by simp [bimap_bimap, id_bimap]
                   /-
                     🎉 no goals
                   -/
                    /-
                      α β : Type u
                      α' β' : Type v
                      F : Type u → Type v → Type w
                      inst✝¹ : Bifunctor F
                      inst✝ : LawfulBifunctor F
                      h : Equiv α β
                      h' : Equiv α' β'
                      x : F β β'
                      ⊢ Eq (Bifunctor.bimap (⇑h) (⇑h') (Bifunctor.bimap (⇑h.symm) (⇑h'.symm) x)) x
                    -/
  right_inv x := by simp [bimap_bimap, id_bimap]
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem mapEquiv_apply (h : α ≃ β) (h' : α' ≃ β') (x : F α α') :
    (mapEquiv F h h' : F α α' ≃ F β β') x = bimap h h' x :=
  rfl


@[simp]
theorem mapEquiv_symm_apply (h : α ≃ β) (h' : α' ≃ β') (y : F β β') :
    (mapEquiv F h h' : F α α' ≃ F β β').symm y = bimap h.symm h'.symm y :=
  rfl


@[simp]
theorem mapEquiv_refl_refl : mapEquiv F (Equiv.refl α) (Equiv.refl α') = Equiv.refl (F α α') := by
  /-
    α : Type u
    α' : Type v
    F : Type u → Type v → Type w
    inst✝¹ : Bifunctor F
    inst✝ : LawfulBifunctor F
    ⊢ Eq (Bifunctor.mapEquiv F (Equiv.refl α) (Equiv.refl α')) (Equiv.refl (F α α'))
  -/
  ext x
  /-
    case H
    α : Type u
    α' : Type v
    F : Type u → Type v → Type w
    inst✝¹ : Bifunctor F
    inst✝ : LawfulBifunctor F
    x : F α α'
    ⊢ Eq ((Bifunctor.mapEquiv F (Equiv.refl α) (Equiv.refl α')) x) ((Equiv.refl (F …
  -/
  simp [id_bimap]
  /-
    🎉 no goals
  -/


