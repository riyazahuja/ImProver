instance [Finite α] [Finite β] : Finite (α × β) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Finite α
    inst✝ : Finite β
    ⊢ Finite (Prod α β)
  -/
  haveI := Fintype.ofFinite α
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Finite α
    inst✝ : Finite β
    this : Fintype α
    ⊢ Finite (Prod α β)
  -/
  haveI := Fintype.ofFinite β
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Finite α
    inst✝ : Finite β
    this✝ : Fintype α
    this : Fintype β
    ⊢ Finite (Prod α β)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {α β : Sort*} [Finite α] [Finite β] : Finite (PProd α β) :=
  of_equiv _ Equiv.pprodEquivProdPLift.symm


theorem prod_left (β) [Finite (α × β)] [Nonempty β] : Finite α :=
  of_surjective (Prod.fst : α × β → α) Prod.fst_surjective


theorem prod_right (α) [Finite (α × β)] [Nonempty α] : Finite β :=
  of_surjective (Prod.snd : α × β → β) Prod.snd_surjective


instance Pi.finite {α : Sort*} {β : α → Sort*} [Finite α] [∀ a, Finite (β a)] :
    Finite (∀ a, β a) := by
  classical
  haveI := Fintype.ofFinite (PLift α)
  haveI := fun a => Fintype.ofFinite (PLift (β a))
  exact
    Finite.of_equiv (∀ a : PLift α, PLift (β (Equiv.plift a)))
      (Equiv.piCongr Equiv.plift fun _ => Equiv.plift)


instance [Finite α] {n : ℕ} : Finite (Sym α n) := by
  classical
  haveI := Fintype.ofFinite α
  infer_instance


instance Function.Embedding.finite {α β : Sort*} [Finite β] : Finite (α ↪ β) := by
  /-
    α✝ : Type u_1
    β✝ : Type u_2
    α : Sort u_3
    β : Sort u_4
    inst✝ : Finite β
    ⊢ Finite (Function.Embedding α β)
  -/
  cases' isEmpty_or_nonempty (α ↪ β) with _ h
  · -- Porting note: infer_instance fails because it applies `Finite.of_fintype` and produces a
    -- "stuck at solving universe constraint" error.
    /-
      case inl
      α✝ : Type u_1
      β✝ : Type u_2
      α : Sort u_3
      β : Sort u_4
      inst✝ : Finite β
      h✝ : IsEmpty (Function.Embedding α β)
      ⊢ Finite (Function.Embedding α β)
    -/
    apply Finite.of_subsingleton
    /-
      🎉 no goals
    -/

    /-
      case inr
      α✝ : Type u_1
      β✝ : Type u_2
      α : Sort u_3
      β : Sort u_4
      inst✝ : Finite β
      h : Nonempty (Function.Embedding α β)
      ⊢ Finite (Function.Embedding α β)
    -/
  · refine h.elim fun f => ?_
    /-
      case inr
      α✝ : Type u_1
      β✝ : Type u_2
      α : Sort u_3
      β : Sort u_4
      inst✝ : Finite β
      h : Nonempty (Function.Embedding α β)
      f : Function.Embedding α β
      ⊢ Finite (Function.Embedding α β)
    -/
    haveI : Finite α := Finite.of_injective _ f.injective
    /-
      case inr
      α✝ : Type u_1
      β✝ : Type u_2
      α : Sort u_3
      β : Sort u_4
      inst✝ : Finite β
      h : Nonempty (Function.Embedding α β)
      f : Function.Embedding α β
      this : Finite α
      ⊢ Finite (Function.Embedding α β)
    -/
    exact Finite.of_injective _ DFunLike.coe_injective
    /-
      🎉 no goals
    -/


instance Equiv.finite_right {α β : Sort*} [Finite β] : Finite (α ≃ β) :=
  Finite.of_injective Equiv.toEmbedding fun e₁ e₂ h => Equiv.ext <| by
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      α : Sort u_3
      β : Sort u_4
      inst✝ : Finite β
      e₁ e₂ : Equiv α β
      h : Eq e₁.toEmbedding e₂.toEmbedding
      ⊢ ∀ (x : α), Eq (e₁ x) (e₂ x)
    -/
    convert DFunLike.congr_fun h using 0
    /-
      🎉 no goals
    -/


instance Equiv.finite_left {α β : Sort*} [Finite α] : Finite (α ≃ β) :=
  Finite.of_equiv _ ⟨Equiv.symm, Equiv.symm, Equiv.symm_symm, Equiv.symm_symm⟩


@[to_additive]
instance MulEquiv.finite_left {α β : Type*} [Mul α] [Mul β] [Finite α] : Finite (α ≃* β) :=
  Finite.of_injective toEquiv toEquiv_injective


@[to_additive]
instance MulEquiv.finite_right {α β : Type*} [Mul α] [Mul β] [Finite β] : Finite (α ≃* β) :=
  Finite.of_injective toEquiv toEquiv_injective


instance fintypeProd (s : Set α) (t : Set β) [Fintype s] [Fintype t] :
    Fintype (s ×ˢ t : Set (α × β)) :=
                                                    /-
                                                      α : Type u_1
                                                      β : Type u_2
                                                      γ : Type u_3
                                                      s : Set α
                                                      t : Set β
                                                      inst✝¹ : Fintype ↑s
                                                      inst✝ : Fintype ↑t
                                                      ⊢ ∀ (x : Prod α β), Iff (Membership.mem (SProd.sprod s.toFinset t.toFinset) x) …
                                                    -/
  Fintype.ofFinset (s.toFinset ×ˢ t.toFinset) <| by simp
                                                    /-
                                                      🎉 no goals
                                                    -/


instance fintypeOffDiag [DecidableEq α] (s : Set α) [Fintype s] : Fintype s.offDiag :=
                                            /-
                                              α : Type u_1
                                              β : Type u_2
                                              γ : Type u_3
                                              inst✝¹ : DecidableEq α
                                              s : Set α
                                              inst✝ : Fintype ↑s
                                              ⊢ ∀ (x : Prod α α), Iff (Membership.mem s.toFinset.offDiag x) (Membership.mem  …
                                            -/
  Fintype.ofFinset s.toFinset.offDiag <| by simp
                                            /-
                                              🎉 no goals
                                            -/


/-- `image2 f s t` is `Fintype` if `s` and `t` are. -/
instance fintypeImage2 [DecidableEq γ] (f : α → β → γ) (s : Set α) (t : Set β) [hs : Fintype s]
    [ht : Fintype t] : Fintype (image2 f s t : Set γ) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : DecidableEq γ
    f : α → β → γ
    s : Set α
    t : Set β
    hs : Fintype ↑s
    ht : Fintype ↑t
    ⊢ Fintype ↑(Set.image2 f s t)
  -/
  rw [← image_prod]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : DecidableEq γ
    f : α → β → γ
    s : Set α
    t : Set β
    hs : Fintype ↑s
    ht : Fintype ↑t
    ⊢ Fintype ↑(Set.image (fun x => f x.1 x.2) (SProd.sprod s t))
  -/
  apply Set.fintypeImage
  /-
    🎉 no goals
  -/


instance finite_prod (s : Set α) (t : Set β) [Finite s] [Finite t] :
    Finite (s ×ˢ t : Set (α × β)) :=
  Finite.of_equiv _ (Equiv.Set.prod s t).symm


instance finite_image2 (f : α → β → γ) (s : Set α) (t : Set β) [Finite s] [Finite t] :
    Finite (image2 f s t : Set γ) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β → γ
    s : Set α
    t : Set β
    inst✝¹ : Finite ↑s
    inst✝ : Finite ↑t
    ⊢ Finite ↑(Set.image2 f s t)
  -/
  rw [← image_prod]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β → γ
    s : Set α
    t : Set β
    inst✝¹ : Finite ↑s
    inst✝ : Finite ↑t
    ⊢ Finite ↑(Set.image (fun x => f x.1 x.2) (SProd.sprod s t))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


protected theorem Finite.prod (hs : s.Finite) (ht : t.Finite) : (s ×ˢ t : Set (α × β)).Finite := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    hs : s.Finite
    ht : t.Finite
    ⊢ (SProd.sprod s t).Finite
  -/
  have := hs.to_subtype
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    hs : s.Finite
    ht : t.Finite
    this : Finite ↑s
    ⊢ (SProd.sprod s t).Finite
  -/
  have := ht.to_subtype
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    hs : s.Finite
    ht : t.Finite
    this✝ : Finite ↑s
    this : Finite ↑t
    ⊢ (SProd.sprod s t).Finite
  -/
  apply toFinite
  /-
    🎉 no goals
  -/


theorem Finite.of_prod_left (h : (s ×ˢ t : Set (α × β)).Finite) : t.Nonempty → s.Finite :=
  fun ⟨b, hb⟩ => (h.image Prod.fst).subset fun a ha => ⟨(a, b), ⟨ha, hb⟩, rfl⟩


theorem Finite.of_prod_right (h : (s ×ˢ t : Set (α × β)).Finite) : s.Nonempty → t.Finite :=
  fun ⟨a, ha⟩ => (h.image Prod.snd).subset fun b hb => ⟨(a, b), ⟨ha, hb⟩, rfl⟩


protected theorem Infinite.prod_left (hs : s.Infinite) (ht : t.Nonempty) : (s ×ˢ t).Infinite :=
  fun h => hs <| h.of_prod_left ht


protected theorem Infinite.prod_right (ht : t.Infinite) (hs : s.Nonempty) : (s ×ˢ t).Infinite :=
  fun h => ht <| h.of_prod_right hs


protected theorem infinite_prod :
    (s ×ˢ t).Infinite ↔ s.Infinite ∧ t.Nonempty ∨ t.Infinite ∧ s.Nonempty := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    ⊢ Iff (SProd.sprod s t).Infinite (Or (And s.Infinite t.Nonempty) (And t.Infini …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      h : (SProd.sprod s t).Infinite
      ⊢ Or (And s.Infinite t.Nonempty) (And t.Infinite s.Nonempty)
    -/
  · simp_rw [Set.Infinite, @and_comm ¬_, ← Classical.not_imp]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      h : (SProd.sprod s t).Infinite
      ⊢ Or (Not (t.Nonempty → s.Finite)) (Not (s.Nonempty → t.Finite))
    -/
    by_contra!
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      h : (SProd.sprod s t).Infinite
      this : And (t.Nonempty → s.Finite) (s.Nonempty → t.Finite)
      ⊢ False
    -/
    exact h ((this.1 h.nonempty.snd).prod <| this.2 h.nonempty.fst)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      ⊢ Or (And s.Infinite t.Nonempty) (And t.Infinite s.Nonempty) → (SProd.sprod s  …
    -/
  · rintro (h | h)
      /-
        case refine_2.inl
        α : Type u_1
        β : Type u_2
        s : Set α
        t : Set β
        h : And s.Infinite t.Nonempty
        ⊢ (SProd.sprod s t).Infinite
      -/
    · exact h.1.prod_left h.2
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        α : Type u_1
        β : Type u_2
        s : Set α
        t : Set β
        h : And t.Infinite s.Nonempty
        ⊢ (SProd.sprod s t).Infinite
      -/
    · exact h.1.prod_right h.2
      /-
        🎉 no goals
      -/


theorem finite_prod : (s ×ˢ t).Finite ↔ (s.Finite ∨ t = ∅) ∧ (t.Finite ∨ s = ∅) := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    ⊢ Iff (SProd.sprod s t).Finite (And (Or s.Finite (Eq t EmptyCollection.emptyCo …
  -/
  simp only [← not_infinite, Set.infinite_prod, not_or, not_and_or, not_nonempty_iff_eq_empty]
  /-
    🎉 no goals
  -/


protected theorem Finite.offDiag {s : Set α} (hs : s.Finite) : s.offDiag.Finite :=
  (hs.prod hs).subset s.offDiag_subset_prod


protected theorem Finite.image2 (f : α → β → γ) (hs : s.Finite) (ht : t.Finite) :
    (image2 f s t).Finite := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Set α
    t : Set β
    f : α → β → γ
    hs : s.Finite
    ht : t.Finite
    ⊢ (Set.image2 f s t).Finite
  -/
  have := hs.to_subtype
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Set α
    t : Set β
    f : α → β → γ
    hs : s.Finite
    ht : t.Finite
    this : Finite ↑s
    ⊢ (Set.image2 f s t).Finite
  -/
  have := ht.to_subtype
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Set α
    t : Set β
    f : α → β → γ
    hs : s.Finite
    ht : t.Finite
    this✝ : Finite ↑s
    this : Finite ↑t
    ⊢ (Set.image2 f s t).Finite
  -/
  apply toFinite
  /-
    🎉 no goals
  -/


theorem Finite.toFinset_prod {s : Set α} {t : Set β} (hs : s.Finite) (ht : t.Finite) :
    hs.toFinset ×ˢ ht.toFinset = (hs.prod ht).toFinset :=
                   /-
                     α : Type u_1
                     β : Type u_2
                     s : Set α
                     t : Set β
                     hs : s.Finite
                     ht : t.Finite
                     ⊢ ∀ (a : Prod α β), Iff (Membership.mem (SProd.sprod hs.toFinset ht.toFinset)  …
                   -/
  Finset.ext <| by simp
                   /-
                     🎉 no goals
                   -/


theorem Finite.toFinset_offDiag {s : Set α} [DecidableEq α] (hs : s.Finite) :
    hs.offDiag.toFinset = hs.toFinset.offDiag :=
                   /-
                     α : Type u_1
                     s : Set α
                     inst✝ : DecidableEq α
                     hs : s.Finite
                     ⊢ ∀ (a : Prod α α), Iff (Membership.mem ⋯.toFinset a) (Membership.mem hs.toFin …
                   -/
  Finset.ext <| by simp
                   /-
                     🎉 no goals
                   -/


theorem finite_image_fst_and_snd_iff {s : Set (α × β)} :
    (Prod.fst '' s).Finite ∧ (Prod.snd '' s).Finite ↔ s.Finite :=
  ⟨fun h => (h.1.prod h.2).subset fun _ h => ⟨mem_image_of_mem _ h, mem_image_of_mem _ h⟩,
    fun h => ⟨h.image _, h.image _⟩⟩


protected theorem Infinite.image2_left (hs : s.Infinite) (hb : b ∈ t)
    (hf : InjOn (fun a => f a b) s) : (image2 f s t).Infinite :=
  (hs.image hf).mono <| image_subset_image2_left hb


protected theorem Infinite.image2_right (ht : t.Infinite) (ha : a ∈ s) (hf : InjOn (f a) t) :
    (image2 f s t).Infinite :=
  (ht.image hf).mono <| image_subset_image2_right ha


theorem infinite_image2 (hfs : ∀ b ∈ t, InjOn (fun a => f a b) s) (hft : ∀ a ∈ s, InjOn (f a) t) :
    (image2 f s t).Infinite ↔ s.Infinite ∧ t.Nonempty ∨ t.Infinite ∧ s.Nonempty := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β → γ
    s : Set α
    t : Set β
    hfs : ∀ (b : β), Membership.mem t b → Set.InjOn (fun a => f a b) s
    hft : ∀ (a : α), Membership.mem s a → Set.InjOn (f a) t
    ⊢ Iff (Set.image2 f s t).Infinite (Or (And s.Infinite t.Nonempty) (And t.Infin …
  -/
  refine ⟨fun h => Set.infinite_prod.1 ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hfs : ∀ (b : β), Membership.mem t b → Set.InjOn (fun a => f a b) s
      hft : ∀ (a : α), Membership.mem s a → Set.InjOn (f a) t
      h : (Set.image2 f s t).Infinite
      ⊢ (SProd.sprod s t).Infinite
    -/
  · rw [← image_uncurry_prod] at h
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hfs : ∀ (b : β), Membership.mem t b → Set.InjOn (fun a => f a b) s
      hft : ∀ (a : α), Membership.mem s a → Set.InjOn (f a) t
      h : (Set.image (Function.uncurry f) (SProd.sprod s t)).Infinite
      ⊢ (SProd.sprod s t).Infinite
    -/
    exact h.of_image _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β → γ
      s : Set α
      t : Set β
      hfs : ∀ (b : β), Membership.mem t b → Set.InjOn (fun a => f a b) s
      hft : ∀ (a : α), Membership.mem s a → Set.InjOn (f a) t
      ⊢ Or (And s.Infinite t.Nonempty) (And t.Infinite s.Nonempty) → (Set.image2 f s …
    -/
  · rintro (⟨hs, b, hb⟩ | ⟨ht, a, ha⟩)
      /-
        case refine_2.inl.intro.intro
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        f : α → β → γ
        s : Set α
        t : Set β
        hfs : ∀ (b : β), Membership.mem t b → Set.InjOn (fun a => f a b) s
        hft : ∀ (a : α), Membership.mem s a → Set.InjOn (f a) t
        hs : s.Infinite
        b : β
        hb : Membership.mem t b
        ⊢ (Set.image2 f s t).Infinite
      -/
    · exact hs.image2_left hb (hfs _ hb)
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr.intro.intro
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        f : α → β → γ
        s : Set α
        t : Set β
        hfs : ∀ (b : β), Membership.mem t b → Set.InjOn (fun a => f a b) s
        hft : ∀ (a : α), Membership.mem s a → Set.InjOn (f a) t
        ht : t.Infinite
        a : α
        ha : Membership.mem s a
        ⊢ (Set.image2 f s t).Infinite
      -/
    · exact ht.image2_right ha (hft _ ha)
      /-
        🎉 no goals
      -/


lemma finite_image2 (hfs : ∀ b ∈ t, InjOn (f · b) s) (hft : ∀ a ∈ s, InjOn (f a) t) :
    (image2 f s t).Finite ↔ s.Finite ∧ t.Finite ∨ s = ∅ ∨ t = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β → γ
    s : Set α
    t : Set β
    hfs : ∀ (b : β), Membership.mem t b → Set.InjOn (fun x => f x b) s
    hft : ∀ (a : α), Membership.mem s a → Set.InjOn (f a) t
    ⊢ Iff (Set.image2 f s t).Finite (Or (And s.Finite t.Finite) (Or (Eq s EmptyCol …
  -/
  rw [← not_infinite, infinite_image2 hfs hft]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β → γ
    s : Set α
    t : Set β
    hfs : ∀ (b : β), Membership.mem t b → Set.InjOn (fun x => f x b) s
    hft : ∀ (a : α), Membership.mem s a → Set.InjOn (f a) t
    ⊢ Iff (Not (Or (And s.Infinite t.Nonempty) (And t.Infinite s.Nonempty))) (Or ( …
  -/
  simp [not_or, -not_and, not_and_or, not_nonempty_iff_eq_empty]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β → γ
    s : Set α
    t : Set β
    hfs : ∀ (b : β), Membership.mem t b → Set.InjOn (fun x => f x b) s
    hft : ∀ (a : α), Membership.mem s a → Set.InjOn (f a) t
    ⊢ Iff (And (Or s.Finite (Eq t EmptyCollection.emptyCollection)) (Or t.Finite ( …
  -/
  aesop
  /-
    🎉 no goals
  -/


