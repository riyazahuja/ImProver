/-- Because `Finset.image` requires a `DecidableEq` instance for the target type, we can only
construct `Functor Finset` when working classically. -/
protected instance functor : Functor Finset where map f s := s.image f


instance lawfulFunctor : LawfulFunctor Finset where
  id_map _ := image_id
  comp_map _ _ _ := image_image.symm
                          /-
                            α✝ β✝ : Type u
                            inst✝ : (P : Prop) → Decidable P
                            α β : Type u_1
                            ⊢ Eq Functor.mapConst (Function.comp Functor.map (Function.const β))
                          -/
  map_const {α} {β} := by simp only [Functor.mapConst, Functor.map]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem fmap_def {s : Finset α} (f : α → β) : f <$> s = s.image f := rfl


protected instance pure : Pure Finset :=
  ⟨fun x => {x}⟩


@[simp]
theorem pure_def {α} : (pure : α → Finset α) = singleton := rfl


protected instance applicative : Applicative Finset :=
  { Finset.functor, Finset.pure with
    seq := fun t s => t.sup fun f => (s ()).image f
    seqLeft := fun s t => if t () = ∅ then ∅ else s
    seqRight := fun s t => if s = ∅ then ∅ else t () }


@[simp]
theorem seq_def (s : Finset α) (t : Finset (α → β)) : t <*> s = t.sup fun f => s.image f :=
  rfl


@[simp]
theorem seqLeft_def (s : Finset α) (t : Finset β) : s <* t = if t = ∅ then ∅ else s :=
  rfl


@[simp]
theorem seqRight_def (s : Finset α) (t : Finset β) : s *> t = if s = ∅ then ∅ else t :=
  rfl


/-- `Finset.image₂` in terms of monadic operations. Note that this can't be taken as the definition
because of the lack of universe polymorphism. -/
theorem image₂_def {α β γ : Type u} (f : α → β → γ) (s : Finset α) (t : Finset β) :
    image₂ f s t = f <$> s <*> t := by
  /-
    inst✝ : (P : Prop) → Decidable P
    α β γ : Type u
    f : α → β → γ
    s : Finset α
    t : Finset β
    ⊢ Eq (Finset.image₂ f s t) (Seq.seq (Functor.map f s) fun x => t)
  -/
  ext
  /-
    case h
    inst✝ : (P : Prop) → Decidable P
    α β γ : Type u
    f : α → β → γ
    s : Finset α
    t : Finset β
    a✝ : γ
    ⊢ Iff (Membership.mem (Finset.image₂ f s t) a✝) (Membership.mem (Seq.seq (Func …
  -/
  simp [mem_sup]
  /-
    🎉 no goals
  -/


instance lawfulApplicative : LawfulApplicative Finset :=
  { Finset.lawfulFunctor with
    seqLeft_eq := fun s t => by
      /-
        α β : Type u
        inst✝ : (P : Prop) → Decidable P
        α✝ β✝ : Type u_1
        s : Finset α✝
        t : Finset β✝
        ⊢ Eq (SeqLeft.seqLeft s fun x => t) (Seq.seq (Functor.map (Function.const β✝)  …
      -/
      rw [seq_def, fmap_def, seqLeft_def]
      /-
        α β : Type u
        inst✝ : (P : Prop) → Decidable P
        α✝ β✝ : Type u_1
        s : Finset α✝
        t : Finset β✝
        ⊢ Eq (ite (Eq t EmptyCollection.emptyCollection) EmptyCollection.emptyCollecti …
      -/
      obtain rfl | ht := t.eq_empty_or_nonempty
        /-
          case inl
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          ⊢ Eq (ite (Eq EmptyCollection.emptyCollection EmptyCollection.emptyCollection) …
        -/
      · simp_rw [image_empty, if_true]
        /-
          case inl
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          ⊢ Eq EmptyCollection.emptyCollection ((Finset.image (Function.const β✝) s).sup …
        -/
        exact (sup_bot _).symm
        /-
          🎉 no goals
        -/
        /-
          case inr
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          t : Finset β✝
          ht : t.Nonempty
          ⊢ Eq (ite (Eq t EmptyCollection.emptyCollection) EmptyCollection.emptyCollecti …
        -/
      · ext a
        /-
          case inr.h
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          t : Finset β✝
          ht : t.Nonempty
          a : α✝
          ⊢ Iff (Membership.mem (ite (Eq t EmptyCollection.emptyCollection) EmptyCollect …
        -/
        rw [if_neg ht.ne_empty, mem_sup]
        /-
          case inr.h
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          t : Finset β✝
          ht : t.Nonempty
          a : α✝
          ⊢ Iff (Membership.mem s a) (Exists fun i => And (Membership.mem (Finset.image  …
        -/
        refine ⟨fun ha => ⟨const _ a, mem_image_of_mem _ ha, mem_image_const_self.2 ht⟩, ?_⟩
        /-
          case inr.h
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          t : Finset β✝
          ht : t.Nonempty
          a : α✝
          ⊢ (Exists fun i => And (Membership.mem (Finset.image (Function.const β✝) s) i) …
        -/
        rintro ⟨f, hf, ha⟩
        /-
          case inr.h.intro.intro
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          t : Finset β✝
          ht : t.Nonempty
          a : α✝
          f : β✝ → α✝
          hf : Membership.mem (Finset.image (Function.const β✝) s) f
          ha : Membership.mem (Finset.image f t) a
          ⊢ Membership.mem s a
        -/
        rw [mem_image] at hf ha
        /-
          case inr.h.intro.intro
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          t : Finset β✝
          ht : t.Nonempty
          a : α✝
          f : β✝ → α✝
          hf : Exists fun a => And (Membership.mem s a) (Eq (Function.const β✝ a) f)
          ha : Exists fun a_1 => And (Membership.mem t a_1) (Eq (f a_1) a)
          ⊢ Membership.mem s a
        -/
        obtain ⟨b, hb, rfl⟩ := hf
        /-
          case inr.h.intro.intro.intro.intro
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          t : Finset β✝
          ht : t.Nonempty
          a b : α✝
          hb : Membership.mem s b
          ha : Exists fun a_1 => And (Membership.mem t a_1) (Eq (Function.const β✝ b a_1 …
          ⊢ Membership.mem s a
        -/
        obtain ⟨_, _, rfl⟩ := ha
        /-
          case inr.h.intro.intro.intro.intro.intro.intro
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          t : Finset β✝
          ht : t.Nonempty
          b : α✝
          hb : Membership.mem s b
          w✝ : β✝
          left✝ : Membership.mem t w✝
          ⊢ Membership.mem s (Function.const β✝ b w✝)
        -/
        exact hb
        /-
          🎉 no goals
        -/
    seqRight_eq := fun s t => by
      /-
        α β : Type u
        inst✝ : (P : Prop) → Decidable P
        α✝ β✝ : Type u_1
        s : Finset α✝
        t : Finset β✝
        ⊢ Eq (SeqRight.seqRight s fun x => t) (Seq.seq (Functor.map (Function.const α✝ …
      -/
      rw [seq_def, fmap_def, seqRight_def]
      /-
        α β : Type u
        inst✝ : (P : Prop) → Decidable P
        α✝ β✝ : Type u_1
        s : Finset α✝
        t : Finset β✝
        ⊢ Eq (ite (Eq s EmptyCollection.emptyCollection) EmptyCollection.emptyCollecti …
      -/
      obtain rfl | hs := s.eq_empty_or_nonempty
        /-
          case inl
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          t : Finset β✝
          ⊢ Eq (ite (Eq EmptyCollection.emptyCollection EmptyCollection.emptyCollection) …
        -/
      · rw [if_pos rfl, image_empty, sup_empty, bot_eq_empty]
        /-
          🎉 no goals
        -/
        /-
          case inr
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          t : Finset β✝
          hs : s.Nonempty
          ⊢ Eq (ite (Eq s EmptyCollection.emptyCollection) EmptyCollection.emptyCollecti …
        -/
      · ext a
        /-
          case inr.h
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          t : Finset β✝
          hs : s.Nonempty
          a : β✝
          ⊢ Iff (Membership.mem (ite (Eq s EmptyCollection.emptyCollection) EmptyCollect …
        -/
        rw [if_neg hs.ne_empty, mem_sup]
        /-
          case inr.h
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          t : Finset β✝
          hs : s.Nonempty
          a : β✝
          ⊢ Iff (Membership.mem t a) (Exists fun i => And (Membership.mem (Finset.image  …
        -/
        refine ⟨fun ha => ⟨id, mem_image_const_self.2 hs, by rwa [image_id]⟩, ?_⟩
        /-
          case inr.h
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          t : Finset β✝
          hs : s.Nonempty
          a : β✝
          ⊢ (Exists fun i => And (Membership.mem (Finset.image (Function.const α✝ id) s) …
        -/
        rintro ⟨f, hf, ha⟩
        /-
          case inr.h.intro.intro
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          t : Finset β✝
          hs : s.Nonempty
          a : β✝
          f : β✝ → β✝
          hf : Membership.mem (Finset.image (Function.const α✝ id) s) f
          ha : Membership.mem (Finset.image f t) a
          ⊢ Membership.mem t a
        -/
        rw [mem_image] at hf ha
        /-
          case inr.h.intro.intro
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          t : Finset β✝
          hs : s.Nonempty
          a : β✝
          f : β✝ → β✝
          hf : Exists fun a => And (Membership.mem s a) (Eq (Function.const α✝ id a) f)
          ha : Exists fun a_1 => And (Membership.mem t a_1) (Eq (f a_1) a)
          ⊢ Membership.mem t a
        -/
        obtain ⟨b, hb, rfl⟩ := ha
        /-
          case inr.h.intro.intro.intro.intro
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          t : Finset β✝
          hs : s.Nonempty
          f : β✝ → β✝
          hf : Exists fun a => And (Membership.mem s a) (Eq (Function.const α✝ id a) f)
          b : β✝
          hb : Membership.mem t b
          ⊢ Membership.mem t (f b)
        -/
        obtain ⟨_, _, rfl⟩ := hf
        /-
          case inr.h.intro.intro.intro.intro.intro.intro
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ : Type u_1
          s : Finset α✝
          t : Finset β✝
          hs : s.Nonempty
          b : β✝
          hb : Membership.mem t b
          w✝ : α✝
          left✝ : Membership.mem s w✝
          ⊢ Membership.mem t (Function.const α✝ id w✝ b)
        -/
        exact hb
        /-
          🎉 no goals
        -/
                              /-
                                α β : Type u
                                inst✝ : (P : Prop) → Decidable P
                                α✝ β✝ : Type u_1
                                f : α✝ → β✝
                                s : Finset α✝
                                ⊢ Eq (Seq.seq (Pure.pure f) fun x => s) (Functor.map f s)
                              -/
    pure_seq := fun f s => by simp only [pure_def, seq_def, sup_singleton, fmap_def]
                              /-
                                🎉 no goals
                              -/
    map_pure := fun _ _ => image_singleton _ _
    seq_pure := fun _ _ => sup_singleton'' _ _
    seq_assoc := fun s t u => by
      /-
        α β : Type u
        inst✝ : (P : Prop) → Decidable P
        α✝ β✝ γ✝ : Type u_1
        s : Finset α✝
        t : Finset (α✝ → β✝)
        u : Finset (β✝ → γ✝)
        ⊢ Eq (Seq.seq u fun x => Seq.seq t fun x => s) (Seq.seq (Seq.seq (Functor.map  …
      -/
      ext a
      /-
        case h
        α β : Type u
        inst✝ : (P : Prop) → Decidable P
        α✝ β✝ γ✝ : Type u_1
        s : Finset α✝
        t : Finset (α✝ → β✝)
        u : Finset (β✝ → γ✝)
        a : γ✝
        ⊢ Iff (Membership.mem (Seq.seq u fun x => Seq.seq t fun x => s) a) (Membership …
      -/
      simp_rw [seq_def, fmap_def]
      /-
        case h
        α β : Type u
        inst✝ : (P : Prop) → Decidable P
        α✝ β✝ γ✝ : Type u_1
        s : Finset α✝
        t : Finset (α✝ → β✝)
        u : Finset (β✝ → γ✝)
        a : γ✝
        ⊢ Iff (Membership.mem (u.sup fun f => Finset.image f (t.sup fun f => Finset.im …
      -/
      simp only [exists_prop, mem_sup, mem_image]
      /-
        case h
        α β : Type u
        inst✝ : (P : Prop) → Decidable P
        α✝ β✝ γ✝ : Type u_1
        s : Finset α✝
        t : Finset (α✝ → β✝)
        u : Finset (β✝ → γ✝)
        a : γ✝
        ⊢ Iff (Exists fun i => And (Membership.mem u i) (Exists fun a_1 => And (Exists …
      -/
      constructor
        /-
          case h.mp
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ γ✝ : Type u_1
          s : Finset α✝
          t : Finset (α✝ → β✝)
          u : Finset (β✝ → γ✝)
          a : γ✝
          ⊢ (Exists fun i => And (Membership.mem u i) (Exists fun a_1 => And (Exists fun …
        -/
      · rintro ⟨g, hg, b, ⟨f, hf, a, ha, rfl⟩, rfl⟩
        /-
          case h.mp.intro.intro.intro.intro.intro.intro.intro.intro
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ γ✝ : Type u_1
          s : Finset α✝
          t : Finset (α✝ → β✝)
          u : Finset (β✝ → γ✝)
          g : β✝ → γ✝
          hg : Membership.mem u g
          f : α✝ → β✝
          hf : Membership.mem t f
          a : α✝
          ha : Membership.mem s a
          ⊢ Exists fun i => And (Exists fun i_1 => And (Exists fun a => And (Membership. …
        -/
        exact ⟨g ∘ f, ⟨comp g, ⟨g, hg, rfl⟩, f, hf, rfl⟩, a, ha, rfl⟩
        /-
          🎉 no goals
        -/
        /-
          case h.mpr
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ γ✝ : Type u_1
          s : Finset α✝
          t : Finset (α✝ → β✝)
          u : Finset (β✝ → γ✝)
          a : γ✝
          ⊢ (Exists fun i => And (Exists fun i_1 => And (Exists fun a => And (Membership …
        -/
      · rintro ⟨c, ⟨_, ⟨g, hg, rfl⟩, f, hf, rfl⟩, a, ha, rfl⟩
        /-
          case h.mpr.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
          α β : Type u
          inst✝ : (P : Prop) → Decidable P
          α✝ β✝ γ✝ : Type u_1
          s : Finset α✝
          t : Finset (α✝ → β✝)
          u : Finset (β✝ → γ✝)
          g : β✝ → γ✝
          hg : Membership.mem u g
          f : α✝ → β✝
          hf : Membership.mem t f
          a : α✝
          ha : Membership.mem s a
          ⊢ Exists fun i => And (Membership.mem u i) (Exists fun a_1 => And (Exists fun  …
        -/
        exact ⟨g, hg, f a, ⟨f, hf, a, ha, rfl⟩, rfl⟩ }
        /-
          🎉 no goals
        -/


instance commApplicative : CommApplicative Finset :=
  { Finset.lawfulApplicative with
    commutative_prod := fun s t => by
      /-
        α β : Type u
        inst✝ : (P : Prop) → Decidable P
        α✝ β✝ : Type u_1
        s : Finset α✝
        t : Finset β✝
        ⊢ Eq (Seq.seq (Functor.map Prod.mk s) fun x => t) (Seq.seq (Functor.map (fun b …
      -/
      simp_rw [seq_def, fmap_def, sup_image, sup_eq_biUnion]
      change (s.biUnion fun a => t.image fun b => (a, b))
        = t.biUnion fun b => s.image fun a => (a, b)
      /-
        α β : Type u
        inst✝ : (P : Prop) → Decidable P
        α✝ β✝ : Type u_1
        s : Finset α✝
        t : Finset β✝
        ⊢ Eq (s.biUnion fun a => Finset.image (fun b => { fst := a, snd := b }) t) (t. …
      -/
      trans s ×ˢ t <;> [rw [product_eq_biUnion]; rw [product_eq_biUnion_right]] }
      /-
        🎉 no goals
      -/


instance : Monad Finset :=
  { Finset.applicative with bind := sup }


@[simp]
theorem bind_def {α β} : (· >>= ·) = sup (α := Finset α) (β := β) :=
  rfl


instance : LawfulMonad Finset :=
  { Finset.lawfulApplicative with
    bind_pure_comp := fun _ _ => sup_singleton'' _ _
    bind_map := fun _ _ => rfl
    pure_bind := fun _ _ => sup_singleton
                                  /-
                                    inst✝ : (P : Prop) → Decidable P
                                    α✝ β✝ γ✝ : Type u_1
                                    s : Finset α✝
                                    f : α✝ → Finset β✝
                                    g : β✝ → Finset γ✝
                                    ⊢ Eq (Bind.bind (Bind.bind s f) g) (Bind.bind s fun x => Bind.bind (f x) g)
                                  -/
    bind_assoc := fun s f g => by simp only [bind, ← sup_biUnion, sup_eq_biUnion, biUnion_biUnion] }
                                  /-
                                    🎉 no goals
                                  -/


instance : Alternative Finset :=
  { Finset.applicative with
    orElse := fun s t => (s ∪ t ())
    failure := ∅ }


/-- Traverse function for `Finset`. -/
def traverse [DecidableEq β] (f : α → F β) (s : Finset α) : F (Finset β) :=
  Multiset.toFinset <$> Multiset.traverse f s.1


@[simp]
theorem id_traverse [DecidableEq α] (s : Finset α) : traverse (pure : α → Id α) s = s := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq (Finset.traverse Pure.pure s) s
  -/
  rw [traverse, Multiset.id_traverse]
  /-
    α : Type u
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq (Functor.map Multiset.toFinset s.val) s
  -/
  exact s.val_toFinset
  /-
    🎉 no goals
  -/


open scoped Classical in
@[simp]
theorem map_comp_coe (h : α → β) :
    Functor.map h ∘ Multiset.toFinset = Multiset.toFinset ∘ Functor.map h :=
  funext fun _ => image_toFinset


open scoped Classical in
@[simp]
theorem map_comp_coe_apply (h : α → β) (s : Multiset α) :
    s.toFinset.image h = (h <$> s).toFinset :=
  congrFun (map_comp_coe h) s


open scoped Classical in
theorem map_traverse (g : α → G β) (h : β → γ) (s : Finset α) :
    Functor.map h <$> traverse g s = traverse (Functor.map h ∘ g) s := by
  /-
    α β γ : Type u
    G : Type u → Type u
    inst✝¹ : Applicative G
    inst✝ : CommApplicative G
    g : α → G β
    h : β → γ
    s : Finset α
    ⊢ Eq (Functor.map (Functor.map h) (Finset.traverse g s)) (Finset.traverse (Fun …
  -/
  unfold traverse
  simp only [Functor.map_map, fmap_def, map_comp_coe_apply, Multiset.fmap_def, ←
    Multiset.map_traverse]


