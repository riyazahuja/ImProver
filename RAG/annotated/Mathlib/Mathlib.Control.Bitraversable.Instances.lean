/-- The bitraverse function for `α × β`. -/
def Prod.bitraverse {α α' β β'} (f : α → F α') (f' : β → F β') : α × β → F (α' × β')
  | (x, y) => Prod.mk <$> f x <*> f' y


instance : Bitraversable Prod where bitraverse := @Prod.bitraverse


instance : LawfulBitraversable Prod := by
  /-
    t : Type u → Type u → Type u
    inst✝¹ : Bitraversable t
    F : Type u → Type u
    inst✝ : Applicative F
    ⊢ LawfulBitraversable Prod
  -/
  constructor <;> intros <;> casesm _ × _ <;>
    /-
      case id_bitraverse.mk
      t : Type u → Type u → Type u
      inst✝¹ : Bitraversable t
      F : Type u → Type u
      inst✝ : Applicative F
      α✝ β✝ : Type u_1
      fst✝ : α✝
      snd✝ : β✝
      ⊢ Eq (Bitraversable.bitraverse Pure.pure Pure.pure { fst := fst✝, snd := snd✝  …
    -/
    /-
      🎉 no goals
    -/
                                                         /-
                                                           🎉 no goals
                                                         -/
                                                         /-
                                                           🎉 no goals
                                                         -/
    simp [bitraverse, Prod.bitraverse, functor_norm] <;> rfl
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- The bitraverse function for `α ⊕ β`. -/
def Sum.bitraverse {α α' β β'} (f : α → F α') (f' : β → F β') : α ⊕ β → F (α' ⊕ β')
  | Sum.inl x => Sum.inl <$> f x
  | Sum.inr x => Sum.inr <$> f' x


instance : Bitraversable Sum where bitraverse := @Sum.bitraverse


instance : LawfulBitraversable Sum := by
  /-
    t : Type u → Type u → Type u
    inst✝¹ : Bitraversable t
    F : Type u → Type u
    inst✝ : Applicative F
    ⊢ LawfulBitraversable Sum
  -/
  constructor <;> intros <;> casesm _ ⊕ _ <;>
    /-
      case id_bitraverse.inl
      t : Type u → Type u → Type u
      inst✝¹ : Bitraversable t
      F : Type u → Type u
      inst✝ : Applicative F
      α✝ β✝ : Type u_1
      val✝ : α✝
      ⊢ Eq (Bitraversable.bitraverse Pure.pure Pure.pure (Sum.inl val✝)) (Pure.pure  …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
                                                        /-
                                                          🎉 no goals
                                                        -/
    simp [bitraverse, Sum.bitraverse, functor_norm] <;> rfl
                                                        /-
                                                          🎉 no goals
                                                        -/



set_option linter.unusedVariables false in
/-- The bitraverse function for `Const`. It throws away the second map. -/
@[nolint unusedArguments]
def Const.bitraverse {F : Type u → Type u} [Applicative F] {α α' β β'} (f : α → F α')
    (f' : β → F β') : Const α β → F (Const α' β') :=
  f


instance Bitraversable.const : Bitraversable Const where bitraverse := @Const.bitraverse


instance LawfulBitraversable.const : LawfulBitraversable Const := by
  /-
    t : Type u → Type u → Type u
    inst✝¹ : Bitraversable t
    F : Type u → Type u
    inst✝ : Applicative F
    ⊢ LawfulBitraversable Functor.Const
  -/
                             /-
                               🎉 no goals
                             -/
                             /-
                               🎉 no goals
                             -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
  constructor <;> intros <;> simp [bitraverse, Const.bitraverse, functor_norm] <;> rfl
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- The bitraverse function for `flip`. -/
nonrec def flip.bitraverse {α α' β β'} (f : α → F α') (f' : β → F β') :
    flip t α β → F (flip t α' β') :=
  (bitraverse f' f : t β α → F (t β' α'))


instance Bitraversable.flip : Bitraversable (flip t) where bitraverse := @flip.bitraverse t _


instance LawfulBitraversable.flip [LawfulBitraversable t] : LawfulBitraversable (flip t) := by
  /-
    t : Type u → Type u → Type u
    inst✝² : Bitraversable t
    F : Type u → Type u
    inst✝¹ : Applicative F
    inst✝ : LawfulBitraversable t
    ⊢ LawfulBitraversable (_root_.flip t)
  -/
                                                              /-
                                                                🎉 no goals
                                                              -/
                                                              /-
                                                                🎉 no goals
                                                              -/
                                                              /-
                                                                🎉 no goals
                                                              -/
  constructor <;> intros <;> casesm LawfulBitraversable t <;> apply_assumption only [*]
                                                              /-
                                                                🎉 no goals
                                                              -/


instance (priority := 10) Bitraversable.traversable {α} : Traversable (t α) where
  traverse := @tsnd t _ _


instance (priority := 10) Bitraversable.isLawfulTraversable [LawfulBitraversable t] {α} :
    LawfulTraversable (t α) := by
  /-
    t : Type u → Type u → Type u
    inst✝² : Bitraversable t
    F : Type u → Type u
    inst✝¹ : Applicative F
    inst✝ : LawfulBitraversable t
    α : Type u
    ⊢ LawfulTraversable (t α)
  -/
  constructor <;> intros <;>
    /-
      case id_traverse
      t : Type u → Type u → Type u
      inst✝² : Bitraversable t
      F : Type u → Type u
      inst✝¹ : Applicative F
      inst✝ : LawfulBitraversable t
      α α✝ : Type u
      x✝ : t α α✝
      ⊢ Eq (Traversable.traverse Pure.pure x✝) x✝
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    simp [traverse, comp_tsnd, functor_norm]
    /-
      case traverse_eq_map_id
      t : Type u → Type u → Type u
      inst✝² : Bitraversable t
      F : Type u → Type u
      inst✝¹ : Applicative F
      inst✝ : LawfulBitraversable t
      α α✝ β✝ : Type u
      f✝ : α✝ → β✝
      x✝ : t α α✝
      ⊢ Eq (Bitraversable.tsnd (Function.comp Pure.pure f✝) x✝) (id.mk (Functor.map  …
    -/
  · simp [tsnd_eq_snd_id, (· <$> ·), id.mk]
    /-
      🎉 no goals
    -/
    /-
      case naturality
      t : Type u → Type u → Type u
      inst✝⁶ : Bitraversable t
      F : Type u → Type u
      inst✝⁵ : Applicative F
      inst✝⁴ : LawfulBitraversable t
      α : Type u
      F✝ G✝ : Type u → Type u
      inst✝³ : Applicative F✝
      inst✝² : Applicative G✝
      inst✝¹ : LawfulApplicative F✝
      inst✝ : LawfulApplicative G✝
      η✝ : ApplicativeTransformation F✝ G✝
      α✝ β✝ : Type u
      f✝ : α✝ → F✝ β✝
      x✝ : t α α✝
      ⊢ Eq (η✝.app (t α β✝) (Bitraversable.tsnd f✝ x✝)) (Bitraversable.tsnd (Functio …
    -/
  · simp [tsnd, binaturality, Function.comp_def, functor_norm]
    /-
      🎉 no goals
    -/


/-- The bitraverse function for `bicompl`. -/
nonrec def Bicompl.bitraverse {m} [Applicative m] {α β α' β'} (f : α → m β) (f' : α' → m β') :
    bicompl t F G α α' → m (bicompl t F G β β') :=
  (bitraverse (traverse f) (traverse f') : t (F α) (G α') → m _)


instance : Bitraversable (bicompl t F G) where bitraverse := @Bicompl.bitraverse t _ F G _ _


instance [LawfulTraversable F] [LawfulTraversable G] [LawfulBitraversable t] :
    LawfulBitraversable (bicompl t F G) := by
  /-
    t : Type u → Type u → Type u
    inst✝⁵ : Bitraversable t
    F G : Type u → Type u
    inst✝⁴ : Traversable F
    inst✝³ : Traversable G
    inst✝² : LawfulTraversable F
    inst✝¹ : LawfulTraversable G
    inst✝ : LawfulBitraversable t
    ⊢ LawfulBitraversable (Function.bicompl t F G)
  -/
  constructor <;> intros <;>
    simp [bitraverse, Bicompl.bitraverse, bimap, traverse_id, bitraverse_id_id, comp_bitraverse,
      functor_norm]
    /-
      case bitraverse_eq_bimap_id
      t : Type u → Type u → Type u
      inst✝⁵ : Bitraversable t
      F G : Type u → Type u
      inst✝⁴ : Traversable F
      inst✝³ : Traversable G
      inst✝² : LawfulTraversable F
      inst✝¹ : LawfulTraversable G
      inst✝ : LawfulBitraversable t
      α✝ α'✝ β✝ β'✝ : Type u
      f✝ : α✝ → β✝
      f'✝ : α'✝ → β'✝
      x✝ : Function.bicompl t F G α✝ α'✝
      ⊢ Eq (Bitraversable.bitraverse (Traversable.traverse (Function.comp Pure.pure  …
    -/
  · simp [traverse_eq_map_id', bitraverse_eq_bimap_id]
    /-
      🎉 no goals
    -/
    /-
      case binaturality
      t : Type u → Type u → Type u
      inst✝⁹ : Bitraversable t
      F G : Type u → Type u
      inst✝⁸ : Traversable F
      inst✝⁷ : Traversable G
      inst✝⁶ : LawfulTraversable F
      inst✝⁵ : LawfulTraversable G
      inst✝⁴ : LawfulBitraversable t
      F✝ G✝ : Type u → Type u
      inst✝³ : Applicative F✝
      inst✝² : Applicative G✝
      inst✝¹ : LawfulApplicative F✝
      inst✝ : LawfulApplicative G✝
      η✝ : ApplicativeTransformation F✝ G✝
      α✝ α'✝ β✝ β'✝ : Type u
      f✝ : α✝ → F✝ β✝
      f'✝ : α'✝ → F✝ β'✝
      x✝ : Function.bicompl t F G α✝ α'✝
      ⊢ Eq (η✝.app (Function.bicompl t F G β✝ β'✝) (Bitraversable.bitraverse (Traver …
    -/
  · dsimp only [bicompl]
    /-
      case binaturality
      t : Type u → Type u → Type u
      inst✝⁹ : Bitraversable t
      F G : Type u → Type u
      inst✝⁸ : Traversable F
      inst✝⁷ : Traversable G
      inst✝⁶ : LawfulTraversable F
      inst✝⁵ : LawfulTraversable G
      inst✝⁴ : LawfulBitraversable t
      F✝ G✝ : Type u → Type u
      inst✝³ : Applicative F✝
      inst✝² : Applicative G✝
      inst✝¹ : LawfulApplicative F✝
      inst✝ : LawfulApplicative G✝
      η✝ : ApplicativeTransformation F✝ G✝
      α✝ α'✝ β✝ β'✝ : Type u
      f✝ : α✝ → F✝ β✝
      f'✝ : α'✝ → F✝ β'✝
      x✝ : Function.bicompl t F G α✝ α'✝
      ⊢ Eq (η✝.app (t (F β✝) (G β'✝)) (Bitraversable.bitraverse (Traversable.travers …
    -/
    simp [binaturality, naturality_pf]
    /-
      🎉 no goals
    -/


/-- The bitraverse function for `bicompr`. -/
nonrec def Bicompr.bitraverse {m} [Applicative m] {α β α' β'} (f : α → m β) (f' : α' → m β') :
    bicompr F t α α' → m (bicompr F t β β') :=
  (traverse (bitraverse f f') : F (t α α') → m _)


instance : Bitraversable (bicompr F t) where bitraverse := @Bicompr.bitraverse t _ F _


instance [LawfulTraversable F] [LawfulBitraversable t] : LawfulBitraversable (bicompr F t) := by
  /-
    t : Type u → Type u → Type u
    inst✝³ : Bitraversable t
    F : Type u → Type u
    inst✝² : Traversable F
    inst✝¹ : LawfulTraversable F
    inst✝ : LawfulBitraversable t
    ⊢ LawfulBitraversable (Function.bicompr F t)
  -/
  constructor <;> intros <;>
    /-
      case id_bitraverse
      t : Type u → Type u → Type u
      inst✝³ : Bitraversable t
      F : Type u → Type u
      inst✝² : Traversable F
      inst✝¹ : LawfulTraversable F
      inst✝ : LawfulBitraversable t
      α✝ β✝ : Type u
      x✝ : Function.bicompr F t α✝ β✝
      ⊢ Eq (Bitraversable.bitraverse Pure.pure Pure.pure x✝) (Pure.pure x✝)
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    simp [bitraverse, Bicompr.bitraverse, bitraverse_id_id, functor_norm]
    /-
      case bitraverse_eq_bimap_id
      t : Type u → Type u → Type u
      inst✝³ : Bitraversable t
      F : Type u → Type u
      inst✝² : Traversable F
      inst✝¹ : LawfulTraversable F
      inst✝ : LawfulBitraversable t
      α✝ α'✝ β✝ β'✝ : Type u
      f✝ : α✝ → β✝
      f'✝ : α'✝ → β'✝
      x✝ : Function.bicompr F t α✝ α'✝
      ⊢ Eq (Traversable.traverse (Bitraversable.bitraverse (Function.comp Pure.pure  …
    -/
  · simp only [bitraverse_eq_bimap_id', traverse_eq_map_id', Function.comp_apply, Id.pure_eq]; rfl
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/
    /-
      case binaturality
      t : Type u → Type u → Type u
      inst✝⁷ : Bitraversable t
      F : Type u → Type u
      inst✝⁶ : Traversable F
      inst✝⁵ : LawfulTraversable F
      inst✝⁴ : LawfulBitraversable t
      F✝ G✝ : Type u → Type u
      inst✝³ : Applicative F✝
      inst✝² : Applicative G✝
      inst✝¹ : LawfulApplicative F✝
      inst✝ : LawfulApplicative G✝
      η✝ : ApplicativeTransformation F✝ G✝
      α✝ α'✝ β✝ β'✝ : Type u
      f✝ : α✝ → F✝ β✝
      f'✝ : α'✝ → F✝ β'✝
      x✝ : Function.bicompr F t α✝ α'✝
      ⊢ Eq (η✝.app (Function.bicompr F t β✝ β'✝) (Traversable.traverse (Bitraversabl …
    -/
  · dsimp only [bicompr]
    /-
      case binaturality
      t : Type u → Type u → Type u
      inst✝⁷ : Bitraversable t
      F : Type u → Type u
      inst✝⁶ : Traversable F
      inst✝⁵ : LawfulTraversable F
      inst✝⁴ : LawfulBitraversable t
      F✝ G✝ : Type u → Type u
      inst✝³ : Applicative F✝
      inst✝² : Applicative G✝
      inst✝¹ : LawfulApplicative F✝
      inst✝ : LawfulApplicative G✝
      η✝ : ApplicativeTransformation F✝ G✝
      α✝ α'✝ β✝ β'✝ : Type u
      f✝ : α✝ → F✝ β✝
      f'✝ : α'✝ → F✝ β'✝
      x✝ : Function.bicompr F t α✝ α'✝
      ⊢ Eq (η✝.app (F (t β✝ β'✝)) (Traversable.traverse (Bitraversable.bitraverse f✝ …
    -/
    simp [naturality, binaturality']
    /-
      🎉 no goals
    -/


