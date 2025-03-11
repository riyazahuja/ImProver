/-- The `Set` functor is a monad.

This is not a global instance because it does not have computational content,
so it does not make much sense using `do` notation in general.
Plus, this would cause monad-related coercions and monad lifting logic to become activated.
Either use `attribute [local instance] Set.monad` to make it be a local instance
or use `SetM.run do ...` when `do` notation is wanted. -/
protected def monad : Monad.{u} Set where
  pure a := {a}
  bind s f := ⋃ i ∈ s, f i
  seq s t := Set.seq s (t ())
  map := Set.image


@[simp]
theorem bind_def : s >>= f = ⋃ i ∈ s, f i :=
  rfl


@[simp]
theorem fmap_eq_image (f : α → β) : f <$> s = f '' s :=
  rfl


@[simp]
theorem seq_eq_set_seq (s : Set (α → β)) (t : Set α) : s <*> t = s.seq t :=
  rfl


@[simp]
theorem pure_def (a : α) : (pure a : Set α) = {a} :=
  rfl


/-- `Set.image2` in terms of monadic operations. Note that this can't be taken as the definition
because of the lack of universe polymorphism. -/
theorem image2_def {α β γ : Type u} (f : α → β → γ) (s : Set α) (t : Set β) :
    image2 f s t = f <$> s <*> t := by
  /-
    α β γ : Type u
    f : α → β → γ
    s : Set α
    t : Set β
    ⊢ Eq (Set.image2 f s t) (Seq.seq (Functor.map f s) fun x => t)
  -/
  ext
  /-
    case h
    α β γ : Type u
    f : α → β → γ
    s : Set α
    t : Set β
    x✝ : γ
    ⊢ Iff (Membership.mem (Set.image2 f s t) x✝) (Membership.mem (Seq.seq (Functor …
  -/
  simp
  /-
    🎉 no goals
  -/


                              /-
                                α β : Type u
                                s : Set α
                                f : α → Set β
                                ⊢ ∀ {α β : Type u_1} (x : α) (y : Set β), Eq (Functor.mapConst x y) (Functor.m …
                              -/
                              /-
                                🎉 no goals
                              -/
                              /-
                                🎉 no goals
                              -/
                                 /-
                                   α β : Type u
                                   s : Set α
                                   f : α → Set β
                                   α✝ β✝ γ✝ : Type u_1
                                   x✝² : Set α✝
                                   x✝¹ : α✝ → Set β✝
                                   x✝ : β✝ → Set γ✝
                                   ⊢ Eq (Bind.bind (Bind.bind x✝² x✝¹) x✝) (Bind.bind x✝² fun x => Bind.bind (x✝¹ …
                                 -/
instance : LawfulMonad Set := LawfulMonad.mk'
                                 /-
                                   🎉 no goals
                                 -/
                              /-
                                🎉 no goals
                              -/
  (id_map := image_id)
  (pure_bind := biUnion_singleton)
  (bind_assoc := fun _ _ _ => by simp only [bind_def, biUnion_iUnion])
  (bind_pure_comp := fun _ _ => (image_eq_iUnion _ _).symm)
  (bind_map := fun _ _ => seq_def.symm)


instance : CommApplicative (Set : Type u → Type u) :=
  ⟨fun s t => prod_image_seq_comm s t⟩


instance : Alternative Set :=
  { Set.monad with
    orElse := fun s t => s ∪ (t ())
    failure := ∅ }


theorem mem_coe_of_mem {a : α} (ha : a ∈ β) (ha' : ⟨a, ha⟩ ∈ γ) : a ∈ (γ : Set α) :=
  ⟨_, ⟨⟨_, rfl⟩, _, ⟨ha', rfl⟩, rfl⟩⟩


theorem coe_subset : (γ : Set α) ⊆ β := by
  /-
    α : Type u
    β : Set α
    γ : Set ↑β
    ⊢ HasSubset.Subset (Bind.bind γ fun a => Pure.pure ↑a) β
  -/
  intro _ ⟨_, ⟨⟨⟨_, ha⟩, rfl⟩, _, ⟨_, rfl⟩, _⟩⟩; convert ha
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem mem_of_mem_coe {a : α} (ha : a ∈ (γ : Set α)) : ⟨a, coe_subset ha⟩ ∈ γ := by
  /-
    α : Type u
    β : Set α
    γ : Set ↑β
    a : α
    ha : Membership.mem (Bind.bind γ fun a => Pure.pure ↑a) a
    ⊢ Membership.mem γ ⟨a, ⋯⟩
  -/
  rcases ha with ⟨_, ⟨_, rfl⟩, _, ⟨ha, rfl⟩, _⟩; convert ha
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem eq_univ_of_coe_eq (hγ : (γ : Set α) = β) : γ = univ :=
  eq_univ_of_forall fun ⟨_, ha⟩ => mem_of_mem_coe <| hγ.symm ▸ ha


theorem image_coe_eq_restrict_image {δ : Type*} {f : α → δ} : f '' γ = β.restrict f '' γ :=
  ext fun _ =>
    ⟨fun ⟨_, h, ha⟩ => ⟨_, mem_of_mem_coe h, ha⟩, fun ⟨_, h, ha⟩ => ⟨_, mem_coe_of_mem _ h, ha⟩⟩


/-- The coercion from `Set.monad` as an instance is equal to the coercion in `Data.Set.Notation`. -/
theorem coe_eq_image_val (t : Set s) :
    @Lean.Internal.coeM Set s α _ Set.monad t = (t : Set α) := by
  /-
    α : Type u
    s : Set α
    t : Set ↑s
    ⊢ Eq (Lean.Internal.coeM t) (Set.image Subtype.val t)
  -/
  change ⋃ (x ∈ t), {x.1} = _
  /-
    α : Type u
    s : Set α
    t : Set ↑s
    ⊢ Eq (Set.iUnion fun x => Set.iUnion fun h => Singleton.singleton ↑x) (Set.ima …
  -/
  ext
  /-
    case h
    α : Type u
    s : Set α
    t : Set ↑s
    x✝ : α
    ⊢ Iff (Membership.mem (Set.iUnion fun x => Set.iUnion fun h => Singleton.singl …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem mem_image_val_of_mem (ha : a ∈ β) (ha' : ⟨a, ha⟩ ∈ γ) : a ∈ (γ : Set α) :=
  ⟨_, ha', rfl⟩


theorem image_val_subset : (γ : Set α) ⊆ β := by
  /-
    α : Type u
    β : Set α
    γ : Set ↑β
    ⊢ HasSubset.Subset (Set.image Subtype.val γ) β
  -/
  rintro _ ⟨⟨_, ha⟩, _, rfl⟩; exact ha
                              /-
                                🎉 no goals
                              -/


theorem mem_of_mem_image_val (ha : a ∈ (γ : Set α)) : ⟨a, image_val_subset ha⟩ ∈ γ := by
  /-
    α : Type u
    β : Set α
    γ : Set ↑β
    a : α
    ha : Membership.mem (Set.image Subtype.val γ) a
    ⊢ Membership.mem γ ⟨a, ⋯⟩
  -/
  rcases ha with ⟨_, ha, rfl⟩; exact ha
                               /-
                                 🎉 no goals
                               -/


theorem eq_univ_of_image_val_eq (hγ : (γ : Set α) = β) : γ = univ :=
  eq_univ_of_forall fun ⟨_, ha⟩ => mem_of_mem_image_val <| hγ.symm ▸ ha


theorem image_image_val_eq_restrict_image {δ : Type*} {f : α → δ} : f '' γ = β.restrict f '' γ := by
  /-
    α : Type u
    β : Set α
    γ : Set ↑β
    δ : Type u_1
    f : α → δ
    ⊢ Eq (Set.image f (Set.image Subtype.val γ)) (Set.image (β.restrict f) γ)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- This is `Set` but with a `Monad` instance. -/
def SetM (α : Type u) := Set α


instance : Monad SetM := Set.monad


/-- Evaluates the `SetM` monad, yielding a `Set`.
Implementation note: this is the identity function. -/
protected def SetM.run {α : Type*} (s : SetM α) : Set α := s

