/-- Lawless bitraversable bifunctor. This only holds data for the bimap and bitraverse. -/
class Bitraversable (t : Type u → Type u → Type u) extends Bifunctor t where
  bitraverse :
    ∀ {m : Type u → Type u} [Applicative m] {α α' β β'},
      (α → m α') → (β → m β') → t α β → m (t α' β')


/-- A bitraversable functor commutes with all applicative functors. -/
def bisequence {t m} [Bitraversable t] [Applicative m] {α β} : t (m α) (m β) → m (t α β) :=
  bitraverse id id


/-- Bifunctor. This typeclass asserts that a lawless bitraversable bifunctor is lawful. -/
class LawfulBitraversable (t : Type u → Type u → Type u) [Bitraversable t] extends
  LawfulBifunctor t : Prop where
  -- Porting note: need to specify `m := Id` because `id` no longer has a `Monad` instance
  id_bitraverse : ∀ {α β} (x : t α β), bitraverse (m := Id) pure pure x = pure x
  comp_bitraverse :
    ∀ {F G} [Applicative F] [Applicative G] [LawfulApplicative F] [LawfulApplicative G]
      {α α' β β' γ γ'} (f : β → F γ) (f' : β' → F γ') (g : α → G β) (g' : α' → G β') (x : t α α'),
      bitraverse (Comp.mk ∘ map f ∘ g) (Comp.mk ∘ map f' ∘ g') x =
        Comp.mk (bitraverse f f' <$> bitraverse g g' x)
  bitraverse_eq_bimap_id :
    ∀ {α α' β β'} (f : α → β) (f' : α' → β') (x : t α α'),
      bitraverse (m := Id) (pure ∘ f) (pure ∘ f') x = pure (bimap f f' x)
  binaturality :
    ∀ {F G} [Applicative F] [Applicative G] [LawfulApplicative F] [LawfulApplicative G]
      (η : ApplicativeTransformation F G) {α α' β β'} (f : α → F β) (f' : α' → F β') (x : t α α'),
      η (bitraverse f f' x) = bitraverse (@η _ ∘ f) (@η _ ∘ f') x


attribute [higher_order bitraverse_id_id] id_bitraverse


attribute [higher_order bitraverse_comp] comp_bitraverse


attribute [higher_order] binaturality bitraverse_eq_bimap_id


