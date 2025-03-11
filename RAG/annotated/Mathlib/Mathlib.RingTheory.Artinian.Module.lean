/-- `IsArtinian R M` is the proposition that `M` is an Artinian `R`-module,
implemented as the well-foundedness of submodule inclusion. -/
abbrev IsArtinian (R M) [Semiring R] [AddCommMonoid M] [Module R M] : Prop :=
  WellFoundedLT (Submodule R M)


theorem isArtinian_iff (R M) [Semiring R] [AddCommMonoid M] [Module R M] : IsArtinian R M ↔
    WellFounded (· < · : Submodule R M → Submodule R M → Prop) :=
  isWellFounded_iff _ _


theorem isArtinian_of_injective (f : M →ₗ[R] P) (h : Function.Injective f) [IsArtinian R P] :
    IsArtinian R M :=
  ⟨Subrelation.wf
    (fun {A B} hAB => show A.map f < B.map f from Submodule.map_strictMono_of_injective h hAB)
    (InvImage.wf (Submodule.map f) IsWellFounded.wf)⟩


instance isArtinian_submodule' [IsArtinian R M] (N : Submodule R M) : IsArtinian R N :=
  isArtinian_of_injective N.subtype Subtype.val_injective


theorem isArtinian_of_le {s t : Submodule R M} [IsArtinian R t] (h : s ≤ t) : IsArtinian R s :=
  isArtinian_of_injective (Submodule.inclusion h) (Submodule.inclusion_injective h)


theorem isArtinian_of_surjective (f : M →ₗ[R] P) (hf : Function.Surjective f) [IsArtinian R M] :
    IsArtinian R P :=
  ⟨Subrelation.wf
    (fun {A B} hAB =>
      show A.comap f < B.comap f from Submodule.comap_strictMono_of_surjective hf hAB)
    (InvImage.wf (Submodule.comap f) IsWellFounded.wf)⟩


instance isArtinian_of_quotient_of_artinian
    (N : Submodule R M) [IsArtinian R M] : IsArtinian R (M ⧸ N) :=
  isArtinian_of_surjective M (Submodule.mkQ N) (Submodule.Quotient.mk_surjective N)


instance isArtinian_range (f : M →ₗ[R] P) [IsArtinian R M] : IsArtinian R (LinearMap.range f) :=
  isArtinian_of_surjective _ _ f.surjective_rangeRestrict


theorem isArtinian_of_linearEquiv (f : M ≃ₗ[R] P) [IsArtinian R M] : IsArtinian R P :=
  isArtinian_of_surjective _ f.toLinearMap f.toEquiv.surjective


theorem LinearEquiv.isArtinian_iff (f : M ≃ₗ[R] P) : IsArtinian R M ↔ IsArtinian R P :=
  ⟨fun _ ↦ isArtinian_of_linearEquiv f, fun _ ↦ isArtinian_of_linearEquiv f.symm⟩


theorem isArtinian_of_range_eq_ker [IsArtinian R M] [IsArtinian R P] (f : M →ₗ[R] N) (g : N →ₗ[R] P)
    (h : LinearMap.range f = LinearMap.ker g) : IsArtinian R N :=
  wellFounded_lt_exact_sequence (LinearMap.range f) (Submodule.map (f.ker.liftQ f le_rfl))
    (Submodule.comap (f.ker.liftQ f le_rfl))
    (Submodule.comap g.rangeRestrict) (Submodule.map g.rangeRestrict)
    (Submodule.gciMapComap <| LinearMap.ker_eq_bot.mp <| Submodule.ker_liftQ_eq_bot _ _ _ le_rfl)
    (Submodule.giMapComap g.surjective_rangeRestrict)
        /-
          R : Type u_1
          M : Type u_2
          P : Type u_3
          N : Type u_4
          inst✝⁸ : Ring R
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : AddCommGroup P
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R M
          inst✝³ : Module R P
          inst✝² : Module R N
          inst✝¹ : IsArtinian R M
          inst✝ : IsArtinian R P
          f : LinearMap (RingHom.id R) M N
          g : LinearMap (RingHom.id R) N P
          h : Eq (LinearMap.range f) (LinearMap.ker g)
          ⊢ ∀ (a : Submodule R N), Eq (Submodule.map ((LinearMap.ker f).liftQ f ⋯) (Subm …
        -/
    (by simp [Submodule.map_comap_eq, inf_comm, Submodule.range_liftQ])
        /-
          🎉 no goals
        -/
        /-
          R : Type u_1
          M : Type u_2
          P : Type u_3
          N : Type u_4
          inst✝⁸ : Ring R
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : AddCommGroup P
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R M
          inst✝³ : Module R P
          inst✝² : Module R N
          inst✝¹ : IsArtinian R M
          inst✝ : IsArtinian R P
          f : LinearMap (RingHom.id R) M N
          g : LinearMap (RingHom.id R) N P
          h : Eq (LinearMap.range f) (LinearMap.ker g)
          ⊢ ∀ (a : Submodule R N), Eq (Submodule.comap g.rangeRestrict (Submodule.map g. …
        -/
    (by simp [Submodule.comap_map_eq, h])
        /-
          🎉 no goals
        -/


theorem isArtinian_iff_submodule_quotient (S : Submodule R P) :
    IsArtinian R P ↔ IsArtinian R S ∧ IsArtinian R (P ⧸ S) := by
  /-
    R : Type u_1
    P : Type u_3
    inst✝² : Ring R
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    S : Submodule R P
    ⊢ Iff (IsArtinian R P) (And (IsArtinian R (Subtype fun x => Membership.mem S x …
  -/
  refine ⟨fun h ↦ ⟨inferInstance, inferInstance⟩, fun ⟨_, _⟩ ↦ ?_⟩
  /-
    R : Type u_1
    P : Type u_3
    inst✝² : Ring R
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    S : Submodule R P
    x✝ : And (IsArtinian R (Subtype fun x => Membership.mem S x)) (IsArtinian R (H …
    left✝ : IsArtinian R (Subtype fun x => Membership.mem S x)
    right✝ : IsArtinian R (HasQuotient.Quotient P S)
    ⊢ IsArtinian R P
  -/
  apply isArtinian_of_range_eq_ker S.subtype S.mkQ
  /-
    R : Type u_1
    P : Type u_3
    inst✝² : Ring R
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    S : Submodule R P
    x✝ : And (IsArtinian R (Subtype fun x => Membership.mem S x)) (IsArtinian R (H …
    left✝ : IsArtinian R (Subtype fun x => Membership.mem S x)
    right✝ : IsArtinian R (HasQuotient.Quotient P S)
    ⊢ Eq (LinearMap.range S.subtype) (LinearMap.ker S.mkQ)
  -/
  rw [Submodule.ker_mkQ, Submodule.range_subtype]
  /-
    🎉 no goals
  -/


instance isArtinian_prod [IsArtinian R M] [IsArtinian R P] : IsArtinian R (M × P) :=
  isArtinian_of_range_eq_ker (LinearMap.inl R M P) (LinearMap.snd R M P) (LinearMap.range_inl R M P)


instance (priority := 100) isArtinian_of_finite [Finite M] : IsArtinian R M :=
  ⟨Finite.wellFounded_of_trans_of_irrefl _⟩

-- Porting note: elab_as_elim can only be global and cannot be changed on an imported decl
-- attribute [local elab_as_elim] Finite.induction_empty_option


instance isArtinian_sup (M₁ M₂ : Submodule R P) [IsArtinian R M₁] [IsArtinian R M₂] :
    IsArtinian R ↥(M₁ ⊔ M₂) := by
  /-
    R : Type u_1
    M : Type u_2
    P : Type u_3
    N : Type u_4
    inst✝⁸ : Ring R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup P
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R M
    inst✝³ : Module R P
    inst✝² : Module R N
    M₁ M₂ : Submodule R P
    inst✝¹ : IsArtinian R (Subtype fun x => Membership.mem M₁ x)
    inst✝ : IsArtinian R (Subtype fun x => Membership.mem M₂ x)
    ⊢ IsArtinian R (Subtype fun x => Membership.mem (Max.max M₁ M₂) x)
  -/
  have := isArtinian_range (M₁.subtype.coprod M₂.subtype)
  /-
    R : Type u_1
    M : Type u_2
    P : Type u_3
    N : Type u_4
    inst✝⁸ : Ring R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup P
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R M
    inst✝³ : Module R P
    inst✝² : Module R N
    M₁ M₂ : Submodule R P
    inst✝¹ : IsArtinian R (Subtype fun x => Membership.mem M₁ x)
    inst✝ : IsArtinian R (Subtype fun x => Membership.mem M₂ x)
    this : IsArtinian R (Subtype fun x => Membership.mem (LinearMap.range (M₁.subt …
    ⊢ IsArtinian R (Subtype fun x => Membership.mem (Max.max M₁ M₂) x)
  -/
  rwa [LinearMap.range_coprod, Submodule.range_subtype, Submodule.range_subtype] at this
  /-
    🎉 no goals
  -/


instance isArtinian_pi :
    ∀ {M : ι → Type*} [∀ i, AddCommGroup (M i)]
      [∀ i, Module R (M i)] [∀ i, IsArtinian R (M i)], IsArtinian R (∀ i, M i) := by
  /-
    R : Type u_1
    M : Type u_2
    P : Type u_3
    N : Type u_4
    inst✝⁷ : Ring R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R P
    inst✝¹ : Module R N
    ι : Type u_5
    inst✝ : Finite ι
    ⊢ ∀ {M : ι → Type u_6} [inst : (i : ι) → AddCommGroup (M i)] [inst_1 : (i : ι) …
  -/
  apply Finite.induction_empty_option _ _ _ ι
    /-
      R : Type u_1
      M : Type u_2
      P : Type u_3
      N : Type u_4
      inst✝⁷ : Ring R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R M
      inst✝² : Module R P
      inst✝¹ : Module R N
      ι : Type u_5
      inst✝ : Finite ι
      ⊢ ∀ {α β : Type u_5}, Equiv α β → (∀ {M : α → Type u_6} [inst : (i : α) → AddC …
    -/
  · exact fun e h ↦ isArtinian_of_linearEquiv (LinearEquiv.piCongrLeft R _ e)
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      M : Type u_2
      P : Type u_3
      N : Type u_4
      inst✝⁷ : Ring R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R M
      inst✝² : Module R P
      inst✝¹ : Module R N
      ι : Type u_5
      inst✝ : Finite ι
      ⊢ ∀ {M : PEmpty.{u_5 + 1} → Type u_6} [inst : (i : PEmpty.{u_5 + 1}) → AddComm …
    -/
  · infer_instance
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      M : Type u_2
      P : Type u_3
      N : Type u_4
      inst✝⁷ : Ring R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R M
      inst✝² : Module R P
      inst✝¹ : Module R N
      ι : Type u_5
      inst✝ : Finite ι
      ⊢ ∀ {α : Type u_5} [inst : Fintype α], (∀ {M : α → Type u_6} [inst : (i : α) → …
    -/
  · exact fun ih ↦ isArtinian_of_linearEquiv (LinearEquiv.piOptionEquivProd R).symm
    /-
      🎉 no goals
    -/


/-- A version of `isArtinian_pi` for non-dependent functions. We need this instance because
sometimes Lean fails to apply the dependent version in non-dependent settings (e.g., it fails to
prove that `ι → ℝ` is finite dimensional over `ℝ`). -/
instance isArtinian_pi' [IsArtinian R M] : IsArtinian R (ι → M) :=
  isArtinian_pi

--Porting note (https://github.com/leanprover-community/mathlib4/issues/10754): new instance

instance isArtinian_finsupp [IsArtinian R M] : IsArtinian R (ι →₀ M) :=
  isArtinian_of_linearEquiv (Finsupp.linearEquivFunOnFinite _ _ _).symm


instance isArtinian_iSup :
    ∀ {M : ι → Submodule R P} [∀ i, IsArtinian R (M i)], IsArtinian R ↥(⨆ i, M i) := by
  /-
    R : Type u_1
    M : Type u_2
    P : Type u_3
    N : Type u_4
    inst✝⁷ : Ring R
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R P
    inst✝¹ : Module R N
    ι : Type u_5
    inst✝ : Finite ι
    ⊢ ∀ {M : ι → Submodule R P} [inst : ∀ (i : ι), IsArtinian R (Subtype fun x =>  …
  -/
  apply Finite.induction_empty_option _ _ _ ι
    /-
      R : Type u_1
      M : Type u_2
      P : Type u_3
      N : Type u_4
      inst✝⁷ : Ring R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R M
      inst✝² : Module R P
      inst✝¹ : Module R N
      ι : Type u_5
      inst✝ : Finite ι
      ⊢ ∀ {α β : Type u_5}, Equiv α β → (∀ {M : α → Submodule R P} [inst : ∀ (i : α) …
    -/
  · intro _ _ e h _ _; rw [← e.iSup_comp]; apply h
                                           /-
                                             🎉 no goals
                                           -/
    /-
      R : Type u_1
      M : Type u_2
      P : Type u_3
      N : Type u_4
      inst✝⁷ : Ring R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R M
      inst✝² : Module R P
      inst✝¹ : Module R N
      ι : Type u_5
      inst✝ : Finite ι
      ⊢ ∀ {M : PEmpty.{u_5 + 1} → Submodule R P} [inst : ∀ (i : PEmpty.{u_5 + 1}), I …
    -/
  · intros; rw [iSup_of_empty]; infer_instance
                                /-
                                  🎉 no goals
                                -/
    /-
      R : Type u_1
      M : Type u_2
      P : Type u_3
      N : Type u_4
      inst✝⁷ : Ring R
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : AddCommGroup N
      inst✝³ : Module R M
      inst✝² : Module R P
      inst✝¹ : Module R N
      ι : Type u_5
      inst✝ : Finite ι
      ⊢ ∀ {α : Type u_5} [inst : Fintype α], (∀ {M : α → Submodule R P} [inst : ∀ (i …
    -/
  · intro _ _ ih _ _; rw [iSup_option]; infer_instance
                                        /-
                                          🎉 no goals
                                        -/


theorem IsArtinian.finite_of_linearIndependent [Nontrivial R] [h : IsArtinian R M] {s : Set M}
    (hs : LinearIndependent R ((↑) : s → M)) : s.Finite := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    h : IsArtinian R M
    s : Set M
    hs : LinearIndependent R Subtype.val
    ⊢ s.Finite
  -/
  refine by_contradiction fun hf => (RelEmbedding.wellFounded_iff_no_descending_seq.1 h.wf).elim' ?_
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    h : IsArtinian R M
    s : Set M
    hs : LinearIndependent R Subtype.val
    hf : Not s.Finite
    ⊢ RelEmbedding (fun x1 x2 => GT.gt x1 x2) fun x1 x2 => LT.lt x1 x2
  -/
  have f : ℕ ↪ s := Set.Infinite.natEmbedding s hf
  have : ∀ n, (↑) ∘ f '' { m | n ≤ m } ⊆ s := by
    rintro n x ⟨y, _, rfl⟩
    exact (f y).2
  have : ∀ a b : ℕ, a ≤ b ↔
      span R (Subtype.val ∘ f '' { m | b ≤ m }) ≤ span R (Subtype.val ∘ f '' { m | a ≤ m }) := by
    intro a b
    rw [span_le_span_iff hs (this b) (this a),
      Set.image_subset_image_iff (Subtype.coe_injective.comp f.injective), Set.subset_def]
    simp only [Set.mem_setOf_eq]
    exact ⟨fun hab x => le_trans hab, fun h => h _ le_rfl⟩
  exact ⟨⟨fun n => span R (Subtype.val ∘ f '' { m | n ≤ m }), fun x y => by
    rw [le_antisymm_iff, ← this y x, ← this x y]
    exact fun ⟨h₁, h₂⟩ => le_antisymm_iff.2 ⟨h₂, h₁⟩⟩, by
    intro a b
    conv_rhs => rw [GT.gt, lt_iff_le_not_le, this, this, ← lt_iff_le_not_le]
    rfl⟩


/-- A module is Artinian iff every nonempty set of submodules has a minimal submodule among them. -/
theorem set_has_minimal_iff_artinian :
    (∀ a : Set <| Submodule R M, a.Nonempty → ∃ M' ∈ a, ∀ I ∈ a, ¬I < M') ↔ IsArtinian R M := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (∀ (a : Set (Submodule R M)), a.Nonempty → Exists fun M' => And (Members …
  -/
  rw [isArtinian_iff, WellFounded.wellFounded_iff_has_min]
  /-
    🎉 no goals
  -/


theorem IsArtinian.set_has_minimal [IsArtinian R M] (a : Set <| Submodule R M) (ha : a.Nonempty) :
    ∃ M' ∈ a, ∀ I ∈ a, ¬I < M' :=
  set_has_minimal_iff_artinian.mpr ‹_› a ha


/-- A module is Artinian iff every decreasing chain of submodules stabilizes. -/
theorem monotone_stabilizes_iff_artinian :
    (∀ f : ℕ →o (Submodule R M)ᵒᵈ, ∃ n, ∀ m, n ≤ m → f n = f m) ↔ IsArtinian R M := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (∀ (f : OrderHom Nat (OrderDual (Submodule R M))), Exists fun n => ∀ (m  …
  -/
  rw [isArtinian_iff]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ⊢ Iff (∀ (f : OrderHom Nat (OrderDual (Submodule R M))), Exists fun n => ∀ (m  …
  -/
  exact WellFounded.monotone_chain_condition.symm
  /-
    🎉 no goals
  -/


theorem monotone_stabilizes (f : ℕ →o (Submodule R M)ᵒᵈ) : ∃ n, ∀ m, n ≤ m → f n = f m :=
  monotone_stabilizes_iff_artinian.mpr ‹_› f


theorem eventuallyConst_of_isArtinian (f : ℕ →o (Submodule R M)ᵒᵈ) :
    atTop.EventuallyConst f := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    f : OrderHom Nat (OrderDual (Submodule R M))
    ⊢ Filter.EventuallyConst (⇑f) Filter.atTop
  -/
  simp_rw [eventuallyConst_atTop, eq_comm]
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    f : OrderHom Nat (OrderDual (Submodule R M))
    ⊢ Exists fun i => ∀ (j : Nat), LE.le i j → Eq (f i) (f j)
  -/
  exact monotone_stabilizes f
  /-
    🎉 no goals
  -/


/-- If `∀ I > J, P I` implies `P J`, then `P` holds for all submodules. -/
theorem induction {P : Submodule R M → Prop} (hgt : ∀ I, (∀ J < I, P J) → P I) (I : Submodule R M) :
    P I :=
  WellFoundedLT.induction I hgt


/-- For any endomorphism of an Artinian module, any sufficiently high iterate has codisjoint kernel
and range. -/
theorem eventually_codisjoint_ker_pow_range_pow (f : M →ₗ[R] M) :
    ∀ᶠ n in atTop, Codisjoint (LinearMap.ker (f ^ n)) (LinearMap.range (f ^ n)) := by
  obtain ⟨n, hn : ∀ m, n ≤ m → LinearMap.range (f ^ n) = LinearMap.range (f ^ m)⟩ :=
    IsArtinian.monotone_stabilizes f.iterateRange
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    f : LinearMap (RingHom.id R) M M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.range (HPow.hPow f n)) (LinearMap. …
    ⊢ Filter.Eventually (fun n => Codisjoint (LinearMap.ker (HPow.hPow f n)) (Line …
  -/
  refine eventually_atTop.mpr ⟨n, fun m hm ↦ codisjoint_iff.mpr ?_⟩
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    f : LinearMap (RingHom.id R) M M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.range (HPow.hPow f n)) (LinearMap. …
    m : Nat
    hm : GE.ge m n
    ⊢ Eq (Max.max (LinearMap.ker (HPow.hPow f m)) (LinearMap.range (HPow.hPow f m) …
  -/
  simp_rw [← hn _ hm, Submodule.eq_top_iff', Submodule.mem_sup]
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    f : LinearMap (RingHom.id R) M M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.range (HPow.hPow f n)) (LinearMap. …
    m : Nat
    hm : GE.ge m n
    ⊢ ∀ (x : M), Exists fun y => And (Membership.mem (LinearMap.ker (HPow.hPow f m …
  -/
  intro x
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    f : LinearMap (RingHom.id R) M M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.range (HPow.hPow f n)) (LinearMap. …
    m : Nat
    hm : GE.ge m n
    x : M
    ⊢ Exists fun y => And (Membership.mem (LinearMap.ker (HPow.hPow f m)) y) (Exis …
  -/
  rsuffices ⟨y, hy⟩ : ∃ y, (f ^ m) ((f ^ n) y) = (f ^ m) x
    /-
      case intro.intro
      R : Type u_1
      M : Type u_2
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsArtinian R M
      f : LinearMap (RingHom.id R) M M
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.range (HPow.hPow f n)) (LinearMap. …
      m : Nat
      hm : GE.ge m n
      x y : M
      hy : Eq ((HPow.hPow f m) ((HPow.hPow f n) y)) ((HPow.hPow f m) x)
      ⊢ Exists fun y => And (Membership.mem (LinearMap.ker (HPow.hPow f m)) y) (Exis …
    -/
  · exact ⟨x - (f ^ n) y, by simp [hy], (f ^ n) y, by simp⟩
    /-
      🎉 no goals
    -/
  -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to change `mem_range` into `mem_range (f := _)`
  simp_rw [f.pow_apply n, f.pow_apply m, ← iterate_add_apply, ← f.pow_apply (m + n),
    ← f.pow_apply m, ← mem_range (f := _), ← hn _ (n.le_add_left m), hn _ hm]
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    f : LinearMap (RingHom.id R) M M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.range (HPow.hPow f n)) (LinearMap. …
    m : Nat
    hm : GE.ge m n
    x : M
    ⊢ Membership.mem (LinearMap.range (HPow.hPow f m)) ((HPow.hPow f m) x)
  -/
  exact LinearMap.mem_range_self (f ^ m) x
  /-
    🎉 no goals
  -/


lemma eventually_iInf_range_pow_eq (f : Module.End R M) :
    ∀ᶠ n in atTop, ⨅ m, LinearMap.range (f ^ m) = LinearMap.range (f ^ n) := by
  obtain ⟨n, hn : ∀ m, n ≤ m → LinearMap.range (f ^ n) = LinearMap.range (f ^ m)⟩ :=
    IsArtinian.monotone_stabilizes f.iterateRange
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    f : Module.End R M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.range (HPow.hPow f n)) (LinearMap. …
    ⊢ Filter.Eventually (fun n => Eq (iInf fun m => LinearMap.range (HPow.hPow f m …
  -/
  refine eventually_atTop.mpr ⟨n, fun l hl ↦ le_antisymm (iInf_le _ _) (le_iInf fun m ↦ ?_)⟩
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    f : Module.End R M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.range (HPow.hPow f n)) (LinearMap. …
    l : Nat
    hl : GE.ge l n
    m : Nat
    ⊢ LE.le (LinearMap.range (HPow.hPow f l)) (LinearMap.range (HPow.hPow f m))
  -/
  rcases le_or_lt l m with h | h
    /-
      case intro.inl
      R : Type u_1
      M : Type u_2
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsArtinian R M
      f : Module.End R M
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.range (HPow.hPow f n)) (LinearMap. …
      l : Nat
      hl : GE.ge l n
      m : Nat
      h : LE.le l m
      ⊢ LE.le (LinearMap.range (HPow.hPow f l)) (LinearMap.range (HPow.hPow f m))
    -/
  · rw [← hn _ (hl.trans h), hn _ hl]
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      R : Type u_1
      M : Type u_2
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsArtinian R M
      f : Module.End R M
      n : Nat
      hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.range (HPow.hPow f n)) (LinearMap. …
      l : Nat
      hl : GE.ge l n
      m : Nat
      h : LT.lt m l
      ⊢ LE.le (LinearMap.range (HPow.hPow f l)) (LinearMap.range (HPow.hPow f m))
    -/
  · exact f.iterateRange.monotone h.le
    /-
      🎉 no goals
    -/


/-- This is the Fitting decomposition of the module `M` with respect to the endomorphism `f`.

See also `LinearMap.isCompl_iSup_ker_pow_iInf_range_pow` for an alternative spelling. -/
theorem eventually_isCompl_ker_pow_range_pow [IsNoetherian R M] (f : M →ₗ[R] M) :
    ∀ᶠ n in atTop, IsCompl (LinearMap.ker (f ^ n)) (LinearMap.range (f ^ n)) := by
  filter_upwards [f.eventually_disjoint_ker_pow_range_pow.and
    f.eventually_codisjoint_ker_pow_range_pow] with n hn
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsArtinian R M
    inst✝ : IsNoetherian R M
    f : LinearMap (RingHom.id R) M M
    n : Nat
    hn : And (Disjoint (LinearMap.ker (HPow.hPow f n)) (LinearMap.range (HPow.hPow …
    ⊢ IsCompl (LinearMap.ker (HPow.hPow f n)) (LinearMap.range (HPow.hPow f n))
  -/
  simpa only [isCompl_iff]
  /-
    🎉 no goals
  -/


/-- This is the Fitting decomposition of the module `M` with respect to the endomorphism `f`.

See also `LinearMap.eventually_isCompl_ker_pow_range_pow` for an alternative spelling. -/
theorem isCompl_iSup_ker_pow_iInf_range_pow [IsNoetherian R M] (f : M →ₗ[R] M) :
    IsCompl (⨆ n, LinearMap.ker (f ^ n)) (⨅ n, LinearMap.range (f ^ n)) := by
  obtain ⟨k, hk⟩ := eventually_atTop.mp <| f.eventually_isCompl_ker_pow_range_pow.and <|
    f.eventually_iInf_range_pow_eq.and f.eventually_iSup_ker_pow_eq
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsArtinian R M
    inst✝ : IsNoetherian R M
    f : LinearMap (RingHom.id R) M M
    k : Nat
    hk : ∀ (b : Nat), GE.ge b k → And (IsCompl (LinearMap.ker (HPow.hPow f b)) (Li …
    ⊢ IsCompl (iSup fun n => LinearMap.ker (HPow.hPow f n)) (iInf fun n => LinearM …
  -/
  obtain ⟨h₁, h₂, h₃⟩ := hk k (le_refl k)
  /-
    case intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsArtinian R M
    inst✝ : IsNoetherian R M
    f : LinearMap (RingHom.id R) M M
    k : Nat
    hk : ∀ (b : Nat), GE.ge b k → And (IsCompl (LinearMap.ker (HPow.hPow f b)) (Li …
    h₁ : IsCompl (LinearMap.ker (HPow.hPow f k)) (LinearMap.range (HPow.hPow f k))
    h₂ : Eq (iInf fun m => LinearMap.range (HPow.hPow f m)) (LinearMap.range (HPow …
    h₃ : Eq (iSup fun m => LinearMap.ker (HPow.hPow f m)) (LinearMap.ker (HPow.hPo …
    ⊢ IsCompl (iSup fun n => LinearMap.ker (HPow.hPow f n)) (iInf fun n => LinearM …
  -/
  rwa [h₂, h₃]
  /-
    🎉 no goals
  -/


/-- Any injective endomorphism of an Artinian module is surjective. -/
theorem surjective_of_injective_endomorphism (f : M →ₗ[R] M) (s : Injective f) : Surjective f := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    f : LinearMap (RingHom.id R) M M
    s : Function.Injective ⇑f
    ⊢ Function.Surjective ⇑f
  -/
  obtain ⟨n, hn⟩ := eventually_atTop.mp f.eventually_codisjoint_ker_pow_range_pow
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    f : LinearMap (RingHom.id R) M M
    s : Function.Injective ⇑f
    n : Nat
    hn : ∀ (b : Nat), GE.ge b n → Codisjoint (LinearMap.ker (HPow.hPow f b)) (Line …
    ⊢ Function.Surjective ⇑f
  -/
  specialize hn (n + 1) (n.le_add_right 1)
  rw [codisjoint_iff, LinearMap.ker_eq_bot.mpr (LinearMap.iterate_injective s _), bot_sup_eq,
    LinearMap.range_eq_top] at hn
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    f : LinearMap (RingHom.id R) M M
    s : Function.Injective ⇑f
    n : Nat
    hn : Function.Surjective ⇑(HPow.hPow f (HAdd.hAdd n 1))
    ⊢ Function.Surjective ⇑f
  -/
  exact LinearMap.surjective_of_iterate_surjective n.succ_ne_zero hn
  /-
    🎉 no goals
  -/


/-- Any injective endomorphism of an Artinian module is bijective. -/
theorem bijective_of_injective_endomorphism (f : M →ₗ[R] M) (s : Injective f) : Bijective f :=
  ⟨s, surjective_of_injective_endomorphism f s⟩


/-- A sequence `f` of submodules of an artinian module,
with the supremum `f (n+1)` and the infimum of `f 0`, ..., `f n` being ⊤,
is eventually ⊤. -/
theorem disjoint_partial_infs_eventually_top (f : ℕ → Submodule R M)
    (h : ∀ n, Disjoint (partialSups (OrderDual.toDual ∘ f) n) (OrderDual.toDual (f (n + 1)))) :
    ∃ n : ℕ, ∀ m, n ≤ m → f m = ⊤ := by
  -- A little off-by-one cleanup first:
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    f : Nat → Submodule R M
    h : ∀ (n : Nat), Disjoint ((partialSups (Function.comp (⇑OrderDual.toDual) f)) …
    ⊢ Exists fun n => ∀ (m : Nat), LE.le n m → Eq (f m) Top.top
  -/
  rsuffices ⟨n, w⟩ : ∃ n : ℕ, ∀ m, n ≤ m → OrderDual.toDual f (m + 1) = ⊤
    /-
      case intro
      R : Type u_1
      M : Type u_2
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsArtinian R M
      f : Nat → Submodule R M
      h : ∀ (n : Nat), Disjoint ((partialSups (Function.comp (⇑OrderDual.toDual) f)) …
      n : Nat
      w : ∀ (m : Nat), LE.le n m → Eq (OrderDual.toDual f (HAdd.hAdd m 1)) Top.top
      ⊢ Exists fun n => ∀ (m : Nat), LE.le n m → Eq (f m) Top.top
    -/
  · use n + 1
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsArtinian R M
      f : Nat → Submodule R M
      h : ∀ (n : Nat), Disjoint ((partialSups (Function.comp (⇑OrderDual.toDual) f)) …
      n : Nat
      w : ∀ (m : Nat), LE.le n m → Eq (OrderDual.toDual f (HAdd.hAdd m 1)) Top.top
      ⊢ ∀ (m : Nat), LE.le (HAdd.hAdd n 1) m → Eq (f m) Top.top
    -/
    rintro (_ | m) p
      /-
        case h.zero
        R : Type u_1
        M : Type u_2
        inst✝³ : Ring R
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : IsArtinian R M
        f : Nat → Submodule R M
        h : ∀ (n : Nat), Disjoint ((partialSups (Function.comp (⇑OrderDual.toDual) f)) …
        n : Nat
        w : ∀ (m : Nat), LE.le n m → Eq (OrderDual.toDual f (HAdd.hAdd m 1)) Top.top
        p : LE.le (HAdd.hAdd n 1) 0
        ⊢ Eq (f 0) Top.top
      -/
    · cases p
      /-
        🎉 no goals
      -/
      /-
        case h.succ
        R : Type u_1
        M : Type u_2
        inst✝³ : Ring R
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : IsArtinian R M
        f : Nat → Submodule R M
        h : ∀ (n : Nat), Disjoint ((partialSups (Function.comp (⇑OrderDual.toDual) f)) …
        n : Nat
        w : ∀ (m : Nat), LE.le n m → Eq (OrderDual.toDual f (HAdd.hAdd m 1)) Top.top
        m : Nat
        p : LE.le (HAdd.hAdd n 1) (HAdd.hAdd m 1)
        ⊢ Eq (f (HAdd.hAdd m 1)) Top.top
      -/
    · apply w
      /-
        case h.succ.a
        R : Type u_1
        M : Type u_2
        inst✝³ : Ring R
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : IsArtinian R M
        f : Nat → Submodule R M
        h : ∀ (n : Nat), Disjoint ((partialSups (Function.comp (⇑OrderDual.toDual) f)) …
        n : Nat
        w : ∀ (m : Nat), LE.le n m → Eq (OrderDual.toDual f (HAdd.hAdd m 1)) Top.top
        m : Nat
        p : LE.le (HAdd.hAdd n 1) (HAdd.hAdd m 1)
        ⊢ LE.le n m
      -/
      exact Nat.succ_le_succ_iff.mp p
      /-
        🎉 no goals
      -/
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    f : Nat → Submodule R M
    h : ∀ (n : Nat), Disjoint ((partialSups (Function.comp (⇑OrderDual.toDual) f)) …
    ⊢ Exists fun n => ∀ (m : Nat), LE.le n m → Eq (OrderDual.toDual f (HAdd.hAdd m …
  -/
  obtain ⟨n, w⟩ := monotone_stabilizes (partialSups (OrderDual.toDual ∘ f))
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    f : Nat → Submodule R M
    h : ∀ (n : Nat), Disjoint ((partialSups (Function.comp (⇑OrderDual.toDual) f)) …
    n : Nat
    w : ∀ (m : Nat), LE.le n m → Eq ((partialSups (Function.comp (⇑OrderDual.toDua …
    ⊢ Exists fun n => ∀ (m : Nat), LE.le n m → Eq (OrderDual.toDual f (HAdd.hAdd m …
  -/
  refine ⟨n, fun m p => ?_⟩
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    f : Nat → Submodule R M
    h : ∀ (n : Nat), Disjoint ((partialSups (Function.comp (⇑OrderDual.toDual) f)) …
    n : Nat
    w : ∀ (m : Nat), LE.le n m → Eq ((partialSups (Function.comp (⇑OrderDual.toDua …
    m : Nat
    p : LE.le n m
    ⊢ Eq (OrderDual.toDual f (HAdd.hAdd m 1)) Top.top
  -/
  exact (h m).eq_bot_of_ge (sup_eq_left.1 <| (w (m + 1) <| le_add_right p).symm.trans <| w m p)
  /-
    🎉 no goals
  -/


theorem range_smul_pow_stabilizes (r : R) :
    ∃ n : ℕ, ∀ m, n ≤ m →
      LinearMap.range (r ^ n • LinearMap.id : M →ₗ[R] M) =
      LinearMap.range (r ^ m • LinearMap.id : M →ₗ[R] M) :=
  monotone_stabilizes
    ⟨fun n => LinearMap.range (r ^ n • LinearMap.id : M →ₗ[R] M), fun n m h x ⟨y, hy⟩ =>
      ⟨r ^ (m - n) • y, by
        /-
          R : Type u_1
          M : Type u_2
          inst✝³ : CommRing R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : IsArtinian R M
          r : R
          n m : Nat
          h : LE.le n m
          x : M
          x✝ : Membership.mem ((fun n => LinearMap.range (HSMul.hSMul (HPow.hPow r n) Li …
          y : M
          hy : Eq ((HSMul.hSMul (HPow.hPow r m) LinearMap.id) y) x
          ⊢ Eq ((HSMul.hSMul (HPow.hPow r n) LinearMap.id) (HSMul.hSMul (HPow.hPow r (HS …
        -/
        dsimp at hy ⊢
        /-
          R : Type u_1
          M : Type u_2
          inst✝³ : CommRing R
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : IsArtinian R M
          r : R
          n m : Nat
          h : LE.le n m
          x : M
          x✝ : Membership.mem ((fun n => LinearMap.range (HSMul.hSMul (HPow.hPow r n) Li …
          y : M
          hy : Eq (HSMul.hSMul (HPow.hPow r m) y) x
          ⊢ Eq (HSMul.hSMul (HPow.hPow r n) (HSMul.hSMul (HPow.hPow r (HSub.hSub m n)) y …
        -/
        rw [← smul_assoc, smul_eq_mul, ← pow_add, ← hy, add_tsub_cancel_of_le h]⟩⟩
        /-
          🎉 no goals
        -/


theorem exists_pow_succ_smul_dvd (r : R) (x : M) :
    ∃ (n : ℕ) (y : M), r ^ n.succ • y = r ^ n • x := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    r : R
    x : M
    ⊢ Exists fun n => Exists fun y => Eq (HSMul.hSMul (HPow.hPow r n.succ) y) (HSM …
  -/
  obtain ⟨n, hn⟩ := IsArtinian.range_smul_pow_stabilizes M r
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    r : R
    x : M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (LinearMap.range (HSMul.hSMul (HPow.hPow r n) …
    ⊢ Exists fun n => Exists fun y => Eq (HSMul.hSMul (HPow.hPow r n.succ) y) (HSM …
  -/
  simp_rw [SetLike.ext_iff] at hn
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsArtinian R M
    r : R
    x : M
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → ∀ (x : M), Iff (Membership.mem (LinearMap.range  …
    ⊢ Exists fun n => Exists fun y => Eq (HSMul.hSMul (HPow.hPow r n.succ) y) (HSM …
  -/
  exact ⟨n, by simpa using hn n.succ n.le_succ (r ^ n • x)⟩
  /-
    🎉 no goals
  -/


theorem isArtinian_of_submodule_of_artinian (R M) [Ring R] [AddCommGroup M] [Module R M]
    (N : Submodule R M) (_ : IsArtinian R M) : IsArtinian R N := inferInstance


/-- If `M / S / R` is a scalar tower, and `M / R` is Artinian, then `M / S` is also Artinian. -/
theorem isArtinian_of_tower (R) {S M} [CommRing R] [Ring S] [AddCommGroup M] [Algebra R S]
    [Module S M] [Module R M] [IsScalarTower R S M] (h : IsArtinian R M) : IsArtinian S M :=
  ⟨(Submodule.restrictScalarsEmbedding R S M).wellFounded h.wf⟩

-- See `Mathlib.RingTheory.Artinian.Ring`

/-- A ring is Artinian if it is Artinian as a module over itself.

Strictly speaking, this should be called `IsLeftArtinianRing` but we omit the `Left` for
convenience in the commutative case. For a right Artinian ring, use `IsArtinian Rᵐᵒᵖ R`.

For equivalent definitions, see `Mathlib.RingTheory.Artinian.Ring`.
-/
@[stacks 00J5]
abbrev IsArtinianRing (R) [Ring R] :=
  IsArtinian R R


theorem isArtinianRing_iff {R} [Ring R] : IsArtinianRing R ↔ IsArtinian R R := Iff.rfl


instance DivisionRing.instIsArtinianRing {K : Type*} [DivisionRing K] : IsArtinianRing K :=
  ⟨Finite.wellFounded_of_trans_of_irrefl _⟩


theorem Ring.isArtinian_of_zero_eq_one {R} [Ring R] (h01 : (0 : R) = 1) : IsArtinianRing R :=
  have := subsingleton_of_zero_eq_one h01
  inferInstance


instance (R) [CommRing R] [IsArtinianRing R] (I : Ideal R) : IsArtinianRing (R ⧸ I) :=
  isArtinian_of_tower R inferInstance


theorem isArtinian_of_fg_of_artinian {R M} [Ring R] [AddCommGroup M] [Module R M]
    (N : Submodule R M) [IsArtinianRing R] (hN : N.FG) : IsArtinian R N := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    N : Submodule R M
    inst✝ : IsArtinianRing R
    hN : N.FG
    ⊢ IsArtinian R (Subtype fun x => Membership.mem N x)
  -/
  let ⟨s, hs⟩ := hN
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    N : Submodule R M
    inst✝ : IsArtinianRing R
    hN : N.FG
    s : Finset M
    hs : Eq (Submodule.span R ↑s) N
    ⊢ IsArtinian R (Subtype fun x => Membership.mem N x)
  -/
  haveI := Classical.decEq M
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    N : Submodule R M
    inst✝ : IsArtinianRing R
    hN : N.FG
    s : Finset M
    hs : Eq (Submodule.span R ↑s) N
    this : DecidableEq M
    ⊢ IsArtinian R (Subtype fun x => Membership.mem N x)
  -/
  haveI := Classical.decEq R
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    N : Submodule R M
    inst✝ : IsArtinianRing R
    hN : N.FG
    s : Finset M
    hs : Eq (Submodule.span R ↑s) N
    this✝ : DecidableEq M
    this : DecidableEq R
    ⊢ IsArtinian R (Subtype fun x => Membership.mem N x)
  -/
  have : ∀ x ∈ s, x ∈ N := fun x hx => hs ▸ Submodule.subset_span hx
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    N : Submodule R M
    inst✝ : IsArtinianRing R
    hN : N.FG
    s : Finset M
    hs : Eq (Submodule.span R ↑s) N
    this✝¹ : DecidableEq M
    this✝ : DecidableEq R
    this : ∀ (x : M), Membership.mem s x → Membership.mem N x
    ⊢ IsArtinian R (Subtype fun x => Membership.mem N x)
  -/
  refine @isArtinian_of_surjective _ ((↑s : Set M) →₀ R) N _ _ _ _ _ ?_ ?_ isArtinian_finsupp
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      N : Submodule R M
      inst✝ : IsArtinianRing R
      hN : N.FG
      s : Finset M
      hs : Eq (Submodule.span R ↑s) N
      this✝¹ : DecidableEq M
      this✝ : DecidableEq R
      this : ∀ (x : M), Membership.mem s x → Membership.mem N x
      ⊢ LinearMap (RingHom.id R) (Finsupp (↑↑s) R) (Subtype fun x => Membership.mem  …
    -/
  · exact Finsupp.linearCombination R (fun i => ⟨i, hs ▸ subset_span i.2⟩)
    /-
      🎉 no goals
    -/
  · rw [← LinearMap.range_eq_top, eq_top_iff,
       ← map_le_map_iff_of_injective (show Injective (Submodule.subtype N)
         from Subtype.val_injective), Submodule.map_top, range_subtype,
         ← Submodule.map_top, ← Submodule.map_comp, Submodule.map_top]
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      N : Submodule R M
      inst✝ : IsArtinianRing R
      hN : N.FG
      s : Finset M
      hs : Eq (Submodule.span R ↑s) N
      this✝¹ : DecidableEq M
      this✝ : DecidableEq R
      this : ∀ (x : M), Membership.mem s x → Membership.mem N x
      ⊢ LE.le N (LinearMap.range (N.subtype.comp (Finsupp.linearCombination R fun i  …
    -/
    subst N
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsArtinianRing R
      s : Finset M
      this✝¹ : DecidableEq M
      this✝ : DecidableEq R
      hN : (Submodule.span R ↑s).FG
      this : ∀ (x : M), Membership.mem s x → Membership.mem (Submodule.span R ↑s) x
      ⊢ LE.le (Submodule.span R ↑s) (LinearMap.range ((Submodule.span R ↑s).subtype. …
    -/
    refine span_le.2 (fun i hi => ?_)
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsArtinianRing R
      s : Finset M
      this✝¹ : DecidableEq M
      this✝ : DecidableEq R
      hN : (Submodule.span R ↑s).FG
      this : ∀ (x : M), Membership.mem s x → Membership.mem (Submodule.span R ↑s) x
      i : M
      hi : Membership.mem (↑s) i
      ⊢ Membership.mem (↑(LinearMap.range ((Submodule.span R ↑s).subtype.comp (Finsu …
    -/
    use Finsupp.single ⟨i, hi⟩ 1
    /-
      case h
      R : Type u_1
      M : Type u_2
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsArtinianRing R
      s : Finset M
      this✝¹ : DecidableEq M
      this✝ : DecidableEq R
      hN : (Submodule.span R ↑s).FG
      this : ∀ (x : M), Membership.mem s x → Membership.mem (Submodule.span R ↑s) x
      i : M
      hi : Membership.mem (↑s) i
      ⊢ Eq (((Submodule.span R ↑s).subtype.comp (Finsupp.linearCombination R fun i = …
    -/
    simp
    /-
      🎉 no goals
    -/


instance isArtinian_of_fg_of_artinian' {R M} [Ring R] [AddCommGroup M] [Module R M]
    [IsArtinianRing R] [Module.Finite R M] : IsArtinian R M :=
  have : IsArtinian R (⊤ : Submodule R M) := isArtinian_of_fg_of_artinian _ Module.Finite.out
  isArtinian_of_linearEquiv (LinearEquiv.ofTop (⊤ : Submodule R M) rfl)


theorem IsArtinianRing.of_finite (R S) [CommRing R] [Ring S] [Algebra R S]
    [IsArtinianRing R] [Module.Finite R S] : IsArtinianRing S :=
  isArtinian_of_tower R isArtinian_of_fg_of_artinian'


/-- In a module over an artinian ring, the submodule generated by finitely many vectors is
artinian. -/
theorem isArtinian_span_of_finite (R) {M} [Ring R] [AddCommGroup M] [Module R M] [IsArtinianRing R]
    {A : Set M} (hA : A.Finite) : IsArtinian R (Submodule.span R A) :=
  isArtinian_of_fg_of_artinian _ (Submodule.fg_def.mpr ⟨A, hA, rfl⟩)


theorem Function.Surjective.isArtinianRing {R} [Ring R] {S} [Ring S] {F}
    [FunLike F R S] [RingHomClass F R S]
    {f : F} (hf : Function.Surjective f) [H : IsArtinianRing R] : IsArtinianRing S := by
  /-
    R : Type u_1
    inst✝³ : Ring R
    S : Type u_2
    inst✝² : Ring S
    F : Type u_3
    inst✝¹ : FunLike F R S
    inst✝ : RingHomClass F R S
    f : F
    hf : Function.Surjective ⇑f
    H : IsArtinianRing R
    ⊢ IsArtinianRing S
  -/
  rw [isArtinianRing_iff] at H ⊢
  /-
    R : Type u_1
    inst✝³ : Ring R
    S : Type u_2
    inst✝² : Ring S
    F : Type u_3
    inst✝¹ : FunLike F R S
    inst✝ : RingHomClass F R S
    f : F
    hf : Function.Surjective ⇑f
    H : IsArtinian R R
    ⊢ IsArtinian S S
  -/
  exact ⟨(Ideal.orderEmbeddingOfSurjective f hf).wellFounded H.wf⟩
  /-
    🎉 no goals
  -/


instance isArtinianRing_range {R} [Ring R] {S} [Ring S] (f : R →+* S) [IsArtinianRing R] :
    IsArtinianRing f.range :=
  f.rangeRestrict_surjective.isArtinianRing


variable (R) in
lemma isField_of_isDomain [IsDomain R] : IsField R := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsArtinianRing R
    inst✝ : IsDomain R
    ⊢ IsField R
  -/
  refine ⟨Nontrivial.exists_pair_ne, mul_comm, fun {x} hx ↦ ?_⟩
  /-
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsArtinianRing R
    inst✝ : IsDomain R
    x : R
    hx : Ne x 0
    ⊢ Exists fun b => Eq (HMul.hMul x b) 1
  -/
  obtain ⟨n, y, hy⟩ := IsArtinian.exists_pow_succ_smul_dvd x (1 : R)
  replace hy : x ^ n * (x * y - 1) = 0 := by
    rw [mul_sub, sub_eq_zero]
    convert hy using 1
    simp [Nat.succ_eq_add_one, pow_add, mul_assoc]
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsArtinianRing R
    inst✝ : IsDomain R
    x : R
    hx : Ne x 0
    n : Nat
    y : R
    hy : Eq (HMul.hMul (HPow.hPow x n) (HSub.hSub (HMul.hMul x y) 1)) 0
    ⊢ Exists fun b => Eq (HMul.hMul x b) 1
  -/
  rw [mul_eq_zero, sub_eq_zero] at hy
  /-
    case intro.intro
    R : Type u_1
    inst✝² : CommRing R
    inst✝¹ : IsArtinianRing R
    inst✝ : IsDomain R
    x : R
    hx : Ne x 0
    n : Nat
    y : R
    hy : Or (Eq (HPow.hPow x n) 0) (Eq (HMul.hMul x y) 1)
    ⊢ Exists fun b => Eq (HMul.hMul x b) 1
  -/
  exact ⟨_, hy.resolve_left <| pow_ne_zero _ hx⟩
  /-
    🎉 no goals
  -/


instance isMaximal_of_isPrime (p : Ideal R) [p.IsPrime] : p.IsMaximal :=
  Ideal.Quotient.maximal_of_isField _ (isField_of_isDomain _)


lemma isPrime_iff_isMaximal (p : Ideal R) : p.IsPrime ↔ p.IsMaximal :=
  ⟨fun _ ↦ isMaximal_of_isPrime p, fun h ↦ h.isPrime⟩


variable (R) in
lemma primeSpectrum_finite : {I : Ideal R | I.IsPrime}.Finite := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsArtinianRing R
    ⊢ (setOf fun I => I.IsPrime).Finite
  -/
  set Spec := {I : Ideal R | I.IsPrime}
  obtain ⟨_, ⟨s, rfl⟩, H⟩ := IsArtinian.set_has_minimal
    (range (Finset.inf · Subtype.val : Finset Spec → Ideal R)) ⟨⊤, ∅, by simp⟩
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsArtinianRing R
    Spec : Set (Ideal R) := setOf fun I => I.IsPrime
    s : Finset ↑Spec
    H : ∀ (I : Submodule R R), Membership.mem (Set.range fun x => x.inf Subtype.va …
    ⊢ Spec.Finite
  -/
  refine Set.finite_def.2 ⟨s, fun p ↦ ?_⟩
  classical
  obtain ⟨q, hq1, hq2⟩ := p.2.inf_le'.mp <| inf_eq_right.mp <|
    inf_le_right.eq_of_not_lt (H (p ⊓ s.inf Subtype.val) ⟨insert p s, by simp⟩)
  rwa [← Subtype.ext <| (@isMaximal_of_isPrime _ _ _ _ q.2).eq_of_le p.2.1 hq2]


@[stacks 00J7]
lemma maximal_ideals_finite : {I : Ideal R | I.IsMaximal}.Finite := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsArtinianRing R
    ⊢ (setOf fun I => I.IsMaximal).Finite
  -/
  simp_rw [← isPrime_iff_isMaximal]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsArtinianRing R
    ⊢ (setOf fun I => I.IsPrime).Finite
  -/
  apply primeSpectrum_finite R
  /-
    🎉 no goals
  -/


@[local instance] lemma subtype_isMaximal_finite : Finite {I : Ideal R | I.IsMaximal} :=
  (maximal_ideals_finite R).to_subtype


/-- A temporary field instance on the quotients by maximal ideals. -/
@[local instance] noncomputable def fieldOfSubtypeIsMaximal
    (I : {I : Ideal R | I.IsMaximal}) : Field (R ⧸ I.1) :=
  have := mem_setOf.mp I.2; Ideal.Quotient.field I.1


/-- The quotient of a commutative artinian ring by its nilradical is isomorphic to
a finite product of fields, namely the quotients by the maximal ideals. -/
noncomputable def quotNilradicalEquivPi :
    R ⧸ nilradical R ≃+* ∀ I : {I : Ideal R | I.IsMaximal}, R ⧸ I.1 :=
  .trans (Ideal.quotEquivOfEq <| ext fun x ↦ by simp_rw [mem_nilradical,
    nilpotent_iff_mem_prime, Submodule.mem_iInf, Subtype.forall, isPrime_iff_isMaximal, mem_setOf])
  (Ideal.quotientInfRingEquivPiQuotient _ fun I J h ↦
                                                                  /-
                                                                    R : Type u_1
                                                                    inst✝¹ : CommRing R
                                                                    inst✝ : IsArtinianRing R
                                                                    I J : ↑(setOf fun I => I.IsMaximal)
                                                                    h : Ne I J
                                                                    ⊢ Ne ↑I ↑J
                                                                  -/
    Ideal.isCoprime_iff_sup_eq.mpr <| I.2.coprime_of_ne J.2 <| by rwa [Ne, Subtype.coe_inj])
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- A reduced commutative artinian ring is isomorphic to a finite product of fields,
namely the quotients by the maximal ideals. -/
noncomputable def equivPi [IsReduced R] : R ≃+* ∀ I : {I : Ideal R | I.IsMaximal}, R ⧸ I.1 :=
  .trans (.symm <| .quotientBot R) <| .trans
    (Ideal.quotEquivOfEq (nilradical_eq_zero R).symm) (quotNilradicalEquivPi R)


