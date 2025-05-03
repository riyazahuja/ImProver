/-- The natural applicative transformation from the identity functor
to `F`, defined by `pure : Π {α}, α → F α`. -/
def PureTransformation :
    ApplicativeTransformation Id F where
  app := @pure F _
  preserves_pure' _ := rfl
  preserves_seq' f x := by
    /-
      t : Type u → Type u
      inst✝⁵ : Traversable t
      inst✝⁴ : LawfulTraversable t
      F G : Type u → Type u
      inst✝³ : Applicative F
      inst✝² : LawfulApplicative F
      inst✝¹ : Applicative G
      inst✝ : LawfulApplicative G
      α β γ : Type u
      g : α → F β
      f✝ : β → γ
      α✝ β✝ : Type u
      f : Id (α✝ → β✝)
      x : Id α✝
      ⊢ Eq (Pure.pure (Seq.seq f fun x_1 => x)) (Seq.seq (Pure.pure f) fun x_1 => Pu …
    -/
    simp only [map_pure, seq_pure]
    /-
      t : Type u → Type u
      inst✝⁵ : Traversable t
      inst✝⁴ : LawfulTraversable t
      F G : Type u → Type u
      inst✝³ : Applicative F
      inst✝² : LawfulApplicative F
      inst✝¹ : Applicative G
      inst✝ : LawfulApplicative G
      α β γ : Type u
      g : α → F β
      f✝ : β → γ
      α✝ β✝ : Type u
      f : Id (α✝ → β✝)
      x : Id α✝
      ⊢ Eq (Pure.pure (Seq.seq f fun x_1 => x)) (Pure.pure (f x))
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem pureTransformation_apply {α} (x : id α) : PureTransformation F x = pure x :=
  rfl


theorem map_eq_traverse_id : map (f := t) f = traverse (m := Id) (pure ∘ f) :=
  funext fun y => (traverse_eq_map_id f y).symm


theorem map_traverse (x : t α) : map f <$> traverse g x = traverse (map f ∘ g) x := by
  /-
    t : Type u → Type u
    inst✝³ : Traversable t
    inst✝² : LawfulTraversable t
    F : Type u → Type u
    inst✝¹ : Applicative F
    inst✝ : LawfulApplicative F
    α β γ : Type u
    g : α → F β
    f : β → γ
    x : t α
    ⊢ Eq (Functor.map (Functor.map f) (Traversable.traverse g x)) (Traversable.tra …
  -/
  rw [map_eq_traverse_id f]
  /-
    t : Type u → Type u
    inst✝³ : Traversable t
    inst✝² : LawfulTraversable t
    F : Type u → Type u
    inst✝¹ : Applicative F
    inst✝ : LawfulApplicative F
    α β γ : Type u
    g : α → F β
    f : β → γ
    x : t α
    ⊢ Eq (Functor.map (Traversable.traverse (Function.comp Pure.pure f)) (Traversa …
  -/
  refine (comp_traverse (pure ∘ f) g x).symm.trans ?_
  /-
    t : Type u → Type u
    inst✝³ : Traversable t
    inst✝² : LawfulTraversable t
    F : Type u → Type u
    inst✝¹ : Applicative F
    inst✝ : LawfulApplicative F
    α β γ : Type u
    g : α → F β
    f : β → γ
    x : t α
    ⊢ Eq (Traversable.traverse (Function.comp Functor.Comp.mk (Function.comp (Func …
  -/
  congr; apply Comp.applicative_comp_id
         /-
           🎉 no goals
         -/


theorem traverse_map (f : β → F γ) (g : α → β) (x : t α) :
    traverse f (g <$> x) = traverse (f ∘ g) x := by
  /-
    t : Type u → Type u
    inst✝³ : Traversable t
    inst✝² : LawfulTraversable t
    F : Type u → Type u
    inst✝¹ : Applicative F
    inst✝ : LawfulApplicative F
    α β γ : Type u
    f : β → F γ
    g : α → β
    x : t α
    ⊢ Eq (Traversable.traverse f (Functor.map g x)) (Traversable.traverse (Functio …
  -/
  rw [@map_eq_traverse_id t _ _ _ _ g]
  /-
    t : Type u → Type u
    inst✝³ : Traversable t
    inst✝² : LawfulTraversable t
    F : Type u → Type u
    inst✝¹ : Applicative F
    inst✝ : LawfulApplicative F
    α β γ : Type u
    f : β → F γ
    g : α → β
    x : t α
    ⊢ Eq (Traversable.traverse f (Traversable.traverse (Function.comp Pure.pure g) …
  -/
  refine (comp_traverse (G := Id) f (pure ∘ g) x).symm.trans ?_
  /-
    t : Type u → Type u
    inst✝³ : Traversable t
    inst✝² : LawfulTraversable t
    F : Type u → Type u
    inst✝¹ : Applicative F
    inst✝ : LawfulApplicative F
    α β γ : Type u
    f : β → F γ
    g : α → β
    x : t α
    ⊢ Eq (Traversable.traverse (Function.comp Functor.Comp.mk (Function.comp (Func …
  -/
  congr; apply Comp.applicative_id_comp
         /-
           🎉 no goals
         -/


theorem pure_traverse (x : t α) : traverse pure x = (pure x : F (t α)) := by
  have : traverse pure x = pure (traverse (m := Id) pure x) :=
      (naturality (PureTransformation F) pure x).symm
  /-
    t : Type u → Type u
    inst✝³ : Traversable t
    inst✝² : LawfulTraversable t
    F : Type u → Type u
    inst✝¹ : Applicative F
    inst✝ : LawfulApplicative F
    α : Type u
    x : t α
    this : Eq (Traversable.traverse Pure.pure x) (Pure.pure (Traversable.traverse  …
    ⊢ Eq (Traversable.traverse Pure.pure x) (Pure.pure x)
  -/
  rwa [id_traverse] at this
  /-
    🎉 no goals
  -/


theorem id_sequence (x : t α) : sequence (f := Id) (pure <$> x) = pure x := by
  /-
    t : Type u → Type u
    inst✝¹ : Traversable t
    inst✝ : LawfulTraversable t
    α : Type u
    x : t α
    ⊢ Eq (sequence (Functor.map Pure.pure x)) (Pure.pure x)
  -/
  simp [sequence, traverse_map, id_traverse]
  /-
    🎉 no goals
  -/


theorem comp_sequence (x : t (F (G α))) :
    sequence (Comp.mk <$> x) = Comp.mk (sequence <$> sequence x) := by
  /-
    t : Type u → Type u
    inst✝⁵ : Traversable t
    inst✝⁴ : LawfulTraversable t
    F G : Type u → Type u
    inst✝³ : Applicative F
    inst✝² : LawfulApplicative F
    inst✝¹ : Applicative G
    inst✝ : LawfulApplicative G
    α : Type u
    x : t (F (G α))
    ⊢ Eq (sequence (Functor.map Functor.Comp.mk x)) (Functor.Comp.mk (Functor.map  …
  -/
  simp only [sequence, traverse_map, id_comp]; rw [← comp_traverse]; simp [map_id]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem naturality' (η : ApplicativeTransformation F G) (x : t (F α)) :
                                                 /-
                                                   t : Type u → Type u
                                                   inst✝⁵ : Traversable t
                                                   inst✝⁴ : LawfulTraversable t
                                                   F G : Type u → Type u
                                                   inst✝³ : Applicative F
                                                   inst✝² : LawfulApplicative F
                                                   inst✝¹ : Applicative G
                                                   inst✝ : LawfulApplicative G
                                                   α : Type u
                                                   η : ApplicativeTransformation F G
                                                   x : t (F α)
                                                   ⊢ Eq ((fun {α} => η.app α) (sequence x)) (sequence (Functor.map (fun {α} => η. …
                                                 -/
    η (sequence x) = sequence (@η _ <$> x) := by simp [sequence, naturality, traverse_map]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[functor_norm]
theorem traverse_id : traverse pure = (pure : t α → Id (t α)) := by
  /-
    t : Type u → Type u
    inst✝¹ : Traversable t
    inst✝ : LawfulTraversable t
    α : Type u
    ⊢ Eq (Traversable.traverse Pure.pure) Pure.pure
  -/
  ext
  /-
    case h
    t : Type u → Type u
    inst✝¹ : Traversable t
    inst✝ : LawfulTraversable t
    α : Type u
    x✝ : t α
    ⊢ Eq (Traversable.traverse Pure.pure x✝) (Pure.pure x✝)
  -/
  exact id_traverse _
  /-
    🎉 no goals
  -/


@[functor_norm]
theorem traverse_comp (g : α → F β) (h : β → G γ) :
    traverse (Comp.mk ∘ map h ∘ g) =
      (Comp.mk ∘ map (traverse h) ∘ traverse g : t α → Comp F G (t γ)) := by
  /-
    t : Type u → Type u
    inst✝⁵ : Traversable t
    inst✝⁴ : LawfulTraversable t
    F G : Type u → Type u
    inst✝³ : Applicative F
    inst✝² : LawfulApplicative F
    inst✝¹ : Applicative G
    inst✝ : LawfulApplicative G
    α β γ : Type u
    g : α → F β
    h : β → G γ
    ⊢ Eq (Traversable.traverse (Function.comp Functor.Comp.mk (Function.comp (Func …
  -/
  ext
  /-
    case h
    t : Type u → Type u
    inst✝⁵ : Traversable t
    inst✝⁴ : LawfulTraversable t
    F G : Type u → Type u
    inst✝³ : Applicative F
    inst✝² : LawfulApplicative F
    inst✝¹ : Applicative G
    inst✝ : LawfulApplicative G
    α β γ : Type u
    g : α → F β
    h : β → G γ
    x✝ : t α
    ⊢ Eq (Traversable.traverse (Function.comp Functor.Comp.mk (Function.comp (Func …
  -/
  exact comp_traverse _ _ _
  /-
    🎉 no goals
  -/


theorem traverse_eq_map_id' (f : β → γ) :
    traverse (m := Id) (pure ∘ f) = pure ∘ (map f : t β → t γ) := by
  /-
    t : Type u → Type u
    inst✝¹ : Traversable t
    inst✝ : LawfulTraversable t
    β γ : Type u
    f : β → γ
    ⊢ Eq (Traversable.traverse (Function.comp Pure.pure f)) (Function.comp Pure.pu …
  -/
  ext
  /-
    case h
    t : Type u → Type u
    inst✝¹ : Traversable t
    inst✝ : LawfulTraversable t
    β γ : Type u
    f : β → γ
    x✝ : t β
    ⊢ Eq (Traversable.traverse (Function.comp Pure.pure f) x✝) (Function.comp Pure …
  -/
  exact traverse_eq_map_id _ _
  /-
    🎉 no goals
  -/

-- @[functor_norm]

theorem traverse_map' (g : α → β) (h : β → G γ) :
    traverse (h ∘ g) = (traverse h ∘ map g : t α → G (t γ)) := by
  /-
    t : Type u → Type u
    inst✝³ : Traversable t
    inst✝² : LawfulTraversable t
    G : Type u → Type u
    inst✝¹ : Applicative G
    inst✝ : LawfulApplicative G
    α β γ : Type u
    g : α → β
    h : β → G γ
    ⊢ Eq (Traversable.traverse (Function.comp h g)) (Function.comp (Traversable.tr …
  -/
  ext
  /-
    case h
    t : Type u → Type u
    inst✝³ : Traversable t
    inst✝² : LawfulTraversable t
    G : Type u → Type u
    inst✝¹ : Applicative G
    inst✝ : LawfulApplicative G
    α β γ : Type u
    g : α → β
    h : β → G γ
    x✝ : t α
    ⊢ Eq (Traversable.traverse (Function.comp h g) x✝) (Function.comp (Traversable …
  -/
  rw [comp_apply, traverse_map]
  /-
    🎉 no goals
  -/


theorem map_traverse' (g : α → G β) (h : β → γ) :
    traverse (map h ∘ g) = (map (map h) ∘ traverse g : t α → G (t γ)) := by
  /-
    t : Type u → Type u
    inst✝³ : Traversable t
    inst✝² : LawfulTraversable t
    G : Type u → Type u
    inst✝¹ : Applicative G
    inst✝ : LawfulApplicative G
    α β γ : Type u
    g : α → G β
    h : β → γ
    ⊢ Eq (Traversable.traverse (Function.comp (Functor.map h) g)) (Function.comp ( …
  -/
  ext
  /-
    case h
    t : Type u → Type u
    inst✝³ : Traversable t
    inst✝² : LawfulTraversable t
    G : Type u → Type u
    inst✝¹ : Applicative G
    inst✝ : LawfulApplicative G
    α β γ : Type u
    g : α → G β
    h : β → γ
    x✝ : t α
    ⊢ Eq (Traversable.traverse (Function.comp (Functor.map h) g) x✝) (Function.com …
  -/
  rw [comp_apply, map_traverse]
  /-
    🎉 no goals
  -/


theorem naturality_pf (η : ApplicativeTransformation F G) (f : α → F β) :
    traverse (@η _ ∘ f) = @η _ ∘ (traverse f : t α → F (t β)) := by
  /-
    t : Type u → Type u
    inst✝⁵ : Traversable t
    inst✝⁴ : LawfulTraversable t
    F G : Type u → Type u
    inst✝³ : Applicative F
    inst✝² : LawfulApplicative F
    inst✝¹ : Applicative G
    inst✝ : LawfulApplicative G
    α β : Type u
    η : ApplicativeTransformation F G
    f : α → F β
    ⊢ Eq (Traversable.traverse (Function.comp (fun {α} => η.app α) f)) (Function.c …
  -/
  ext
  /-
    case h
    t : Type u → Type u
    inst✝⁵ : Traversable t
    inst✝⁴ : LawfulTraversable t
    F G : Type u → Type u
    inst✝³ : Applicative F
    inst✝² : LawfulApplicative F
    inst✝¹ : Applicative G
    inst✝ : LawfulApplicative G
    α β : Type u
    η : ApplicativeTransformation F G
    f : α → F β
    x✝ : t α
    ⊢ Eq (Traversable.traverse (Function.comp (fun {α} => η.app α) f) x✝) (Functio …
  -/
  rw [comp_apply, naturality]
  /-
    🎉 no goals
  -/


