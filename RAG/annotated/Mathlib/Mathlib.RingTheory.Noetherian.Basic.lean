theorem isNoetherian_of_surjective (f : M →ₗ[R] P) (hf : LinearMap.range f = ⊤) [IsNoetherian R M] :
    IsNoetherian R P :=
  ⟨fun s =>
    have : (s.comap f).map f = s := Submodule.map_comap_eq_self <| hf.symm ▸ le_top
    this ▸ (noetherian _).map _⟩


instance isNoetherian_range (f : M →ₗ[R] P) [IsNoetherian R M] :
    IsNoetherian R (LinearMap.range f) :=
  isNoetherian_of_surjective _ _ f.range_rangeRestrict


instance isNoetherian_quotient {A M : Type*} [Ring A] [AddCommGroup M] [SMul R A] [Module R M]
    [Module A M] [IsScalarTower R A M] (N : Submodule A M) [IsNoetherian R M] :
    IsNoetherian R (M ⧸ N) :=
  isNoetherian_of_surjective M ((Submodule.mkQ N).restrictScalars R) <|
    LinearMap.range_eq_top.mpr N.mkQ_surjective


@[deprecated (since := "2024-04-27"), nolint defLemma]
alias Submodule.Quotient.isNoetherian := isNoetherian_quotient


theorem isNoetherian_of_linearEquiv (f : M ≃ₗ[R] P) [IsNoetherian R M] : IsNoetherian R P :=
  isNoetherian_of_surjective _ f.toLinearMap f.range


theorem LinearEquiv.isNoetherian_iff (f : M ≃ₗ[R] P) : IsNoetherian R M ↔ IsNoetherian R P :=
  ⟨fun _ ↦ isNoetherian_of_linearEquiv f, fun _ ↦ isNoetherian_of_linearEquiv f.symm⟩


theorem isNoetherian_top_iff : IsNoetherian R (⊤ : Submodule R M) ↔ IsNoetherian R M :=
  Submodule.topEquiv.isNoetherian_iff


theorem isNoetherian_of_injective [IsNoetherian R P] (f : M →ₗ[R] P) (hf : Function.Injective f) :
    IsNoetherian R M :=
  isNoetherian_of_linearEquiv (LinearEquiv.ofInjective f hf).symm


theorem fg_of_injective [IsNoetherian R P] {N : Submodule R M} (f : M →ₗ[R] P)
    (hf : Function.Injective f) : N.FG :=
  haveI := isNoetherian_of_injective f hf
  IsNoetherian.noetherian N


instance (priority := 80) _root_.isNoetherian_of_finite [Finite M] : IsNoetherian R M :=
                                               /-
                                                 R : Type u_1
                                                 M : Type u_2
                                                 N : Type u_3
                                                 inst✝⁵ : Semiring R
                                                 inst✝⁴ : AddCommMonoid M
                                                 inst✝³ : AddCommMonoid N
                                                 inst✝² : Module R M
                                                 inst✝¹ : Module R N
                                                 inst✝ : Finite M
                                                 s : Submodule R M
                                                 ⊢ Eq (Submodule.span R ↑⋯.toFinset) s
                                               -/
  ⟨fun s => ⟨(s : Set M).toFinite.toFinset, by rw [Set.Finite.coe_toFinset, Submodule.span_eq]⟩⟩
                                               /-
                                                 🎉 no goals
                                               -/

-- see Note [lower instance priority]

instance (priority := 100) IsNoetherian.finite [IsNoetherian R M] : Module.Finite R M :=
  ⟨IsNoetherian.noetherian ⊤⟩


instance {R₁ S : Type*} [CommSemiring R₁] [Semiring S] [Algebra R₁ S]
    [IsNoetherian R₁ S] (I : Ideal S) : Module.Finite R₁ I :=
  IsNoetherian.finite R₁ ((I : Submodule S S).restrictScalars R₁)


theorem Finite.of_injective [IsNoetherian R N] (f : M →ₗ[R] N) (hf : Function.Injective f) :
    Module.Finite R M :=
  ⟨fg_of_injective f hf⟩


theorem isNoetherian_of_ker_bot [IsNoetherian R P] (f : M →ₗ[R] P) (hf : LinearMap.ker f = ⊥) :
    IsNoetherian R M :=
  isNoetherian_of_linearEquiv (LinearEquiv.ofInjective f <| LinearMap.ker_eq_bot.mp hf).symm


theorem fg_of_ker_bot [IsNoetherian R P] {N : Submodule R M} (f : M →ₗ[R] P)
    (hf : LinearMap.ker f = ⊥) : N.FG :=
  haveI := isNoetherian_of_ker_bot f hf
  IsNoetherian.noetherian N


instance isNoetherian_prod [IsNoetherian R M] [IsNoetherian R P] : IsNoetherian R (M × P) :=
  ⟨fun s =>
    Submodule.fg_of_fg_map_of_fg_inf_ker (LinearMap.snd R M P) (noetherian _) <|
      have : s ⊓ LinearMap.ker (LinearMap.snd R M P) ≤ LinearMap.range (LinearMap.inl R M P) :=
        fun x ⟨_, hx2⟩ => ⟨x.1, Prod.ext rfl <| Eq.symm <| LinearMap.mem_ker.1 hx2⟩
      Submodule.map_comap_eq_self this ▸ (noetherian _).map _⟩


instance isNoetherian_sup (M₁ M₂ : Submodule R P) [IsNoetherian R M₁] [IsNoetherian R M₂] :
    IsNoetherian R ↥(M₁ ⊔ M₂) := by
  /-
    R : Type u_1
    M : Type u_2
    P : Type u_3
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup P
    inst✝³ : Module R M
    inst✝² : Module R P
    M₁ M₂ : Submodule R P
    inst✝¹ : IsNoetherian R (Subtype fun x => Membership.mem M₁ x)
    inst✝ : IsNoetherian R (Subtype fun x => Membership.mem M₂ x)
    ⊢ IsNoetherian R (Subtype fun x => Membership.mem (Max.max M₁ M₂) x)
  -/
  have := isNoetherian_range (M₁.subtype.coprod M₂.subtype)
  /-
    R : Type u_1
    M : Type u_2
    P : Type u_3
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup P
    inst✝³ : Module R M
    inst✝² : Module R P
    M₁ M₂ : Submodule R P
    inst✝¹ : IsNoetherian R (Subtype fun x => Membership.mem M₁ x)
    inst✝ : IsNoetherian R (Subtype fun x => Membership.mem M₂ x)
    this : IsNoetherian R (Subtype fun x => Membership.mem (LinearMap.range (M₁.su …
    ⊢ IsNoetherian R (Subtype fun x => Membership.mem (Max.max M₁ M₂) x)
  -/
  rwa [LinearMap.range_coprod, Submodule.range_subtype, Submodule.range_subtype] at this
  /-
    🎉 no goals
  -/


instance isNoetherian_pi :
    ∀ {M : ι → Type*} [∀ i, AddCommGroup (M i)]
      [∀ i, Module R (M i)] [∀ i, IsNoetherian R (M i)], IsNoetherian R (∀ i, M i) := by
  /-
    R : Type u_1
    M : Type u_2
    P : Type u_3
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R P
    ι : Type u_4
    inst✝ : Finite ι
    ⊢ ∀ {M : ι → Type u_5} [inst : (i : ι) → AddCommGroup (M i)] [inst_1 : (i : ι) …
  -/
  apply Finite.induction_empty_option _ _ _ ι
    /-
      R : Type u_1
      M : Type u_2
      P : Type u_3
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R P
      ι : Type u_4
      inst✝ : Finite ι
      ⊢ ∀ {α β : Type u_4}, Equiv α β → (∀ {M : α → Type u_5} [inst : (i : α) → AddC …
    -/
  · exact fun e h ↦ isNoetherian_of_linearEquiv (LinearEquiv.piCongrLeft R _ e)
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      M : Type u_2
      P : Type u_3
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R P
      ι : Type u_4
      inst✝ : Finite ι
      ⊢ ∀ {M : PEmpty.{u_4 + 1} → Type u_5} [inst : (i : PEmpty.{u_4 + 1}) → AddComm …
    -/
  · infer_instance
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      M : Type u_2
      P : Type u_3
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R P
      ι : Type u_4
      inst✝ : Finite ι
      ⊢ ∀ {α : Type u_4} [inst : Fintype α], (∀ {M : α → Type u_5} [inst : (i : α) → …
    -/
  · exact fun ih ↦ isNoetherian_of_linearEquiv (LinearEquiv.piOptionEquivProd R).symm
    /-
      🎉 no goals
    -/


/-- A version of `isNoetherian_pi` for non-dependent functions. We need this instance because
sometimes Lean fails to apply the dependent version in non-dependent settings (e.g., it fails to
prove that `ι → ℝ` is finite dimensional over `ℝ`). -/
instance isNoetherian_pi' [IsNoetherian R M] : IsNoetherian R (ι → M) :=
  isNoetherian_pi


instance isNoetherian_iSup :
    ∀ {M : ι → Submodule R P} [∀ i, IsNoetherian R (M i)], IsNoetherian R ↥(⨆ i, M i) := by
  /-
    R : Type u_1
    M : Type u_2
    P : Type u_3
    inst✝⁵ : Ring R
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup P
    inst✝² : Module R M
    inst✝¹ : Module R P
    ι : Type u_4
    inst✝ : Finite ι
    ⊢ ∀ {M : ι → Submodule R P} [inst : ∀ (i : ι), IsNoetherian R (Subtype fun x = …
  -/
  apply Finite.induction_empty_option _ _ _ ι
    /-
      R : Type u_1
      M : Type u_2
      P : Type u_3
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R P
      ι : Type u_4
      inst✝ : Finite ι
      ⊢ ∀ {α β : Type u_4}, Equiv α β → (∀ {M : α → Submodule R P} [inst : ∀ (i : α) …
    -/
  · intro _ _ e h _ _; rw [← e.iSup_comp]; apply h
                                           /-
                                             🎉 no goals
                                           -/
    /-
      R : Type u_1
      M : Type u_2
      P : Type u_3
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R P
      ι : Type u_4
      inst✝ : Finite ι
      ⊢ ∀ {M : PEmpty.{u_4 + 1} → Submodule R P} [inst : ∀ (i : PEmpty.{u_4 + 1}), I …
    -/
  · intros; rw [iSup_of_empty]; infer_instance
                                /-
                                  🎉 no goals
                                -/
    /-
      R : Type u_1
      M : Type u_2
      P : Type u_3
      inst✝⁵ : Ring R
      inst✝⁴ : AddCommGroup M
      inst✝³ : AddCommGroup P
      inst✝² : Module R M
      inst✝¹ : Module R P
      ι : Type u_4
      inst✝ : Finite ι
      ⊢ ∀ {α : Type u_4} [inst : Fintype α], (∀ {M : α → Submodule R P} [inst : ∀ (i …
    -/
  · intro _ _ ih _ _; rw [iSup_option]; infer_instance
                                        /-
                                          🎉 no goals
                                        -/


instance isNoetherian_linearMap_pi {ι : Type*} [Finite ι] : IsNoetherian R ((ι → R) →ₗ[R] M) :=
  let _i : Fintype ι := Fintype.ofFinite ι; isNoetherian_of_linearEquiv (Module.piEquiv ι R M)


instance isNoetherian_linearMap : IsNoetherian R (N →ₗ[R] M) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : IsNoetherian R M
    inst✝ : Module.Finite R N
    ⊢ IsNoetherian R (LinearMap (RingHom.id R) N M)
  -/
  obtain ⟨n, f, hf⟩ := Module.Finite.exists_fin' R N
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : IsNoetherian R M
    inst✝ : Module.Finite R N
    n : Nat
    f : LinearMap (RingHom.id R) (Fin n → R) N
    hf : Function.Surjective ⇑f
    ⊢ IsNoetherian R (LinearMap (RingHom.id R) N M)
  -/
  let g : (N →ₗ[R] M) →ₗ[R] (Fin n → R) →ₗ[R] M := (LinearMap.llcomp R (Fin n → R) N M).flip f
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁶ : CommRing R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : AddCommGroup N
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : IsNoetherian R M
    inst✝ : Module.Finite R N
    n : Nat
    f : LinearMap (RingHom.id R) (Fin n → R) N
    hf : Function.Surjective ⇑f
    g : LinearMap (RingHom.id R) (LinearMap (RingHom.id R) N M) (LinearMap (RingHo …
    ⊢ IsNoetherian R (LinearMap (RingHom.id R) N M)
  -/
  exact isNoetherian_of_injective g hf.injective_linearMapComp_right
  /-
    🎉 no goals
  -/


/-- If `∀ I > J, P I` implies `P J`, then `P` holds for all submodules. -/
theorem IsNoetherian.induction [IsNoetherian R M] {P : Submodule R M → Prop}
    (hgt : ∀ I, (∀ J > I, P J) → P I) (I : Submodule R M) : P I :=
  IsWellFounded.induction _ I hgt


lemma Submodule.finite_ne_bot_of_iSupIndep {ι : Type*} {N : ι → Submodule R M}
    (h : iSupIndep N) :
    Set.Finite {i | N i ≠ ⊥} :=
  WellFoundedGT.finite_ne_bot_of_iSupIndep h


@[deprecated (since := "2024-11-24")]
alias Submodule.finite_ne_bot_of_independent := Submodule.finite_ne_bot_of_iSupIndep


/-- A linearly-independent family of vectors in a module over a non-trivial ring must be finite if
the module is Noetherian. -/
theorem LinearIndependent.finite_of_isNoetherian [Nontrivial R] {ι} {v : ι → M}
    (hv : LinearIndependent R v) : Finite ι := by
  refine WellFoundedGT.finite_of_iSupIndep
    hv.iSupIndep_span_singleton
    fun i contra => ?_
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsNoetherian R M
    inst✝ : Nontrivial R
    ι : Type u_4
    v : ι → M
    hv : LinearIndependent R v
    i : ι
    contra : Eq (Submodule.span R (Singleton.singleton (v i))) Bot.bot
    ⊢ False
  -/
  apply hv.ne_zero i
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsNoetherian R M
    inst✝ : Nontrivial R
    ι : Type u_4
    v : ι → M
    hv : LinearIndependent R v
    i : ι
    contra : Eq (Submodule.span R (Singleton.singleton (v i))) Bot.bot
    ⊢ Eq (v i) 0
  -/
  have : v i ∈ R ∙ v i := Submodule.mem_span_singleton_self (v i)
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsNoetherian R M
    inst✝ : Nontrivial R
    ι : Type u_4
    v : ι → M
    hv : LinearIndependent R v
    i : ι
    contra : Eq (Submodule.span R (Singleton.singleton (v i))) Bot.bot
    this : Membership.mem (Submodule.span R (Singleton.singleton (v i))) (v i)
    ⊢ Eq (v i) 0
  -/
  rwa [contra, Submodule.mem_bot] at this
  /-
    🎉 no goals
  -/


theorem LinearIndependent.set_finite_of_isNoetherian [Nontrivial R] {s : Set M}
    (hi : LinearIndependent R ((↑) : s → M)) : s.Finite :=
  @Set.toFinite _ _ hi.finite_of_isNoetherian


/-- If the first and final modules in an exact sequence are Noetherian,
  then the middle module is also Noetherian. -/
theorem isNoetherian_of_range_eq_ker [IsNoetherian R P]
    (f : M →ₗ[R] N) (g : N →ₗ[R] P) (h : LinearMap.range f = LinearMap.ker g) :
    IsNoetherian R N :=
  isNoetherian_mk <|
    wellFounded_gt_exact_sequence
      (LinearMap.range f)
      (Submodule.map (f.ker.liftQ f le_rfl))
      (Submodule.comap (f.ker.liftQ f le_rfl))
      (Submodule.comap g.rangeRestrict) (Submodule.map g.rangeRestrict)
      (Submodule.gciMapComap <| LinearMap.ker_eq_bot.mp <| Submodule.ker_liftQ_eq_bot _ _ _ le_rfl)
      (Submodule.giMapComap g.surjective_rangeRestrict)
          /-
            R : Type u_1
            M : Type u_2
            P : Type u_3
            N : Type w
            inst✝⁸ : Ring R
            inst✝⁷ : AddCommGroup M
            inst✝⁶ : Module R M
            inst✝⁵ : AddCommGroup N
            inst✝⁴ : Module R N
            inst✝³ : AddCommGroup P
            inst✝² : Module R P
            inst✝¹ : IsNoetherian R M
            inst✝ : IsNoetherian R P
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
            N : Type w
            inst✝⁸ : Ring R
            inst✝⁷ : AddCommGroup M
            inst✝⁶ : Module R M
            inst✝⁵ : AddCommGroup N
            inst✝⁴ : Module R N
            inst✝³ : AddCommGroup P
            inst✝² : Module R P
            inst✝¹ : IsNoetherian R M
            inst✝ : IsNoetherian R P
            f : LinearMap (RingHom.id R) M N
            g : LinearMap (RingHom.id R) N P
            h : Eq (LinearMap.range f) (LinearMap.ker g)
            ⊢ ∀ (a : Submodule R N), Eq (Submodule.comap g.rangeRestrict (Submodule.map g. …
          -/
      (by simp [Submodule.comap_map_eq, h])
          /-
            🎉 no goals
          -/


theorem isNoetherian_iff_submodule_quotient (S : Submodule R P) :
    IsNoetherian R P ↔ IsNoetherian R S ∧ IsNoetherian R (P ⧸ S) := by
  /-
    R : Type u_1
    P : Type u_3
    inst✝² : Ring R
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    S : Submodule R P
    ⊢ Iff (IsNoetherian R P) (And (IsNoetherian R (Subtype fun x => Membership.mem …
  -/
  refine ⟨fun _ ↦ ⟨inferInstance, inferInstance⟩, fun ⟨_, _⟩ ↦ ?_⟩
  /-
    R : Type u_1
    P : Type u_3
    inst✝² : Ring R
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    S : Submodule R P
    x✝ : And (IsNoetherian R (Subtype fun x => Membership.mem S x)) (IsNoetherian  …
    left✝ : IsNoetherian R (Subtype fun x => Membership.mem S x)
    right✝ : IsNoetherian R (HasQuotient.Quotient P S)
    ⊢ IsNoetherian R P
  -/
  apply isNoetherian_of_range_eq_ker S.subtype S.mkQ
  /-
    R : Type u_1
    P : Type u_3
    inst✝² : Ring R
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    S : Submodule R P
    x✝ : And (IsNoetherian R (Subtype fun x => Membership.mem S x)) (IsNoetherian  …
    left✝ : IsNoetherian R (Subtype fun x => Membership.mem S x)
    right✝ : IsNoetherian R (HasQuotient.Quotient P S)
    ⊢ Eq (LinearMap.range S.subtype) (LinearMap.ker S.mkQ)
  -/
  rw [Submodule.ker_mkQ, Submodule.range_subtype]
  /-
    🎉 no goals
  -/


/-- A sequence `f` of submodules of a noetherian module,
with `f (n+1)` disjoint from the supremum of `f 0`, ..., `f n`,
is eventually zero. -/
theorem IsNoetherian.disjoint_partialSups_eventually_bot
    (f : ℕ → Submodule R M) (h : ∀ n, Disjoint (partialSups f n) (f (n + 1))) :
    ∃ n : ℕ, ∀ m, n ≤ m → f m = ⊥ := by
  -- A little off-by-one cleanup first:
  suffices t : ∃ n : ℕ, ∀ m, n ≤ m → f (m + 1) = ⊥ by
    obtain ⟨n, w⟩ := t
    use n + 1
    rintro (_ | m) p
    · cases p
    · apply w
      exact Nat.succ_le_succ_iff.mp p
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsNoetherian R M
    f : Nat → Submodule R M
    h : ∀ (n : Nat), Disjoint ((partialSups f) n) (f (HAdd.hAdd n 1))
    ⊢ Exists fun n => ∀ (m : Nat), LE.le n m → Eq (f (HAdd.hAdd m 1)) Bot.bot
  -/
  obtain ⟨n, w⟩ := monotone_stabilizes_iff_noetherian.mpr inferInstance (partialSups f)
  exact
    ⟨n, fun m p =>
      (h m).eq_bot_of_ge <| sup_eq_left.1 <| (w (m + 1) <| le_add_right p).symm.trans <| w m p⟩


/-- Modules over the trivial ring are Noetherian. -/
instance (priority := 100) isNoetherian_of_subsingleton (R M) [Subsingleton R] [Semiring R]
    [AddCommMonoid M] [Module R M] : IsNoetherian R M :=
  haveI := Module.subsingleton R M
  isNoetherian_of_finite R M


theorem isNoetherian_of_submodule_of_noetherian (R M) [Semiring R] [AddCommMonoid M] [Module R M]
    (N : Submodule R M) (h : IsNoetherian R M) : IsNoetherian R N :=
  isNoetherian_mk ⟨OrderEmbedding.wellFounded (Submodule.MapSubtype.orderEmbedding N).dual h.wf⟩


/-- If `M / S / R` is a scalar tower, and `M / R` is Noetherian, then `M / S` is
also noetherian. -/
theorem isNoetherian_of_tower (R) {S M} [Semiring R] [Semiring S] [AddCommMonoid M] [SMul R S]
    [Module S M] [Module R M] [IsScalarTower R S M] (h : IsNoetherian R M) : IsNoetherian S M :=
  isNoetherian_mk ⟨(Submodule.restrictScalarsEmbedding R S M).dual.wellFounded h.wf⟩


theorem isNoetherian_of_fg_of_noetherian {R M} [Ring R] [AddCommGroup M] [Module R M]
    (N : Submodule R M) [I : IsNoetherianRing R] (hN : N.FG) : IsNoetherian R N := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    N : Submodule R M
    I : IsNoetherianRing R
    hN : N.FG
    ⊢ IsNoetherian R (Subtype fun x => Membership.mem N x)
  -/
  let ⟨s, hs⟩ := hN
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    N : Submodule R M
    I : IsNoetherianRing R
    hN : N.FG
    s : Finset M
    hs : Eq (Submodule.span R ↑s) N
    ⊢ IsNoetherian R (Subtype fun x => Membership.mem N x)
  -/
  haveI := Classical.decEq M
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    N : Submodule R M
    I : IsNoetherianRing R
    hN : N.FG
    s : Finset M
    hs : Eq (Submodule.span R ↑s) N
    this : DecidableEq M
    ⊢ IsNoetherian R (Subtype fun x => Membership.mem N x)
  -/
  haveI := Classical.decEq R
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    N : Submodule R M
    I : IsNoetherianRing R
    hN : N.FG
    s : Finset M
    hs : Eq (Submodule.span R ↑s) N
    this✝ : DecidableEq M
    this : DecidableEq R
    ⊢ IsNoetherian R (Subtype fun x => Membership.mem N x)
  -/
  have : ∀ x ∈ s, x ∈ N := fun x hx => hs ▸ Submodule.subset_span hx
  refine
    @isNoetherian_of_surjective
      R ((↑s : Set M) → R) N _ _ _ (Pi.module _ _ _) _ ?_ ?_ isNoetherian_pi
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      N : Submodule R M
      I : IsNoetherianRing R
      hN : N.FG
      s : Finset M
      hs : Eq (Submodule.span R ↑s) N
      this✝¹ : DecidableEq M
      this✝ : DecidableEq R
      this : ∀ (x : M), Membership.mem s x → Membership.mem N x
      ⊢ LinearMap (RingHom.id R) (↑↑s → R) (Subtype fun x => Membership.mem N x)
    -/
  · fapply LinearMap.mk
      /-
        case refine_1.toAddHom
        R : Type u_1
        M : Type u_2
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        N : Submodule R M
        I : IsNoetherianRing R
        hN : N.FG
        s : Finset M
        hs : Eq (Submodule.span R ↑s) N
        this✝¹ : DecidableEq M
        this✝ : DecidableEq R
        this : ∀ (x : M), Membership.mem s x → Membership.mem N x
        ⊢ AddHom (↑↑s → R) (Subtype fun x => Membership.mem N x)
      -/
    · fapply AddHom.mk
        /-
          case refine_1.toAddHom.toFun
          R : Type u_1
          M : Type u_2
          inst✝² : Ring R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          N : Submodule R M
          I : IsNoetherianRing R
          hN : N.FG
          s : Finset M
          hs : Eq (Submodule.span R ↑s) N
          this✝¹ : DecidableEq M
          this✝ : DecidableEq R
          this : ∀ (x : M), Membership.mem s x → Membership.mem N x
          ⊢ (↑↑s → R) → Subtype fun x => Membership.mem N x
        -/
      · exact fun f => ⟨∑ i ∈ s.attach, f i • i.1, N.sum_mem fun c _ => N.smul_mem _ <| this _ c.2⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_1.toAddHom.map_add'
          R : Type u_1
          M : Type u_2
          inst✝² : Ring R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          N : Submodule R M
          I : IsNoetherianRing R
          hN : N.FG
          s : Finset M
          hs : Eq (Submodule.span R ↑s) N
          this✝¹ : DecidableEq M
          this✝ : DecidableEq R
          this : ∀ (x : M), Membership.mem s x → Membership.mem N x
          ⊢ ∀ (x y : ↑↑s → R), Eq ⟨s.attach.sum fun i => HSMul.hSMul (HAdd.hAdd x y i) ↑ …
        -/
      · intro f g
        /-
          case refine_1.toAddHom.map_add'
          R : Type u_1
          M : Type u_2
          inst✝² : Ring R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          N : Submodule R M
          I : IsNoetherianRing R
          hN : N.FG
          s : Finset M
          hs : Eq (Submodule.span R ↑s) N
          this✝¹ : DecidableEq M
          this✝ : DecidableEq R
          this : ∀ (x : M), Membership.mem s x → Membership.mem N x
          f g : ↑↑s → R
          ⊢ Eq ⟨s.attach.sum fun i => HSMul.hSMul (HAdd.hAdd f g i) ↑i, ⋯⟩ (HAdd.hAdd ⟨s …
        -/
        apply Subtype.eq
        /-
          case refine_1.toAddHom.map_add'.a
          R : Type u_1
          M : Type u_2
          inst✝² : Ring R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          N : Submodule R M
          I : IsNoetherianRing R
          hN : N.FG
          s : Finset M
          hs : Eq (Submodule.span R ↑s) N
          this✝¹ : DecidableEq M
          this✝ : DecidableEq R
          this : ∀ (x : M), Membership.mem s x → Membership.mem N x
          f g : ↑↑s → R
          ⊢ Eq ↑⟨s.attach.sum fun i => HSMul.hSMul (HAdd.hAdd f g i) ↑i, ⋯⟩ ↑(HAdd.hAdd  …
        -/
        change (∑ i ∈ s.attach, (f i + g i) • _) = _
        /-
          case refine_1.toAddHom.map_add'.a
          R : Type u_1
          M : Type u_2
          inst✝² : Ring R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          N : Submodule R M
          I : IsNoetherianRing R
          hN : N.FG
          s : Finset M
          hs : Eq (Submodule.span R ↑s) N
          this✝¹ : DecidableEq M
          this✝ : DecidableEq R
          this : ∀ (x : M), Membership.mem s x → Membership.mem N x
          f g : ↑↑s → R
          ⊢ Eq (s.attach.sum fun i => HSMul.hSMul (HAdd.hAdd (f i) (g i)) ↑i) ↑(HAdd.hAd …
        -/
        simp only [add_smul, Finset.sum_add_distrib]
        /-
          case refine_1.toAddHom.map_add'.a
          R : Type u_1
          M : Type u_2
          inst✝² : Ring R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          N : Submodule R M
          I : IsNoetherianRing R
          hN : N.FG
          s : Finset M
          hs : Eq (Submodule.span R ↑s) N
          this✝¹ : DecidableEq M
          this✝ : DecidableEq R
          this : ∀ (x : M), Membership.mem s x → Membership.mem N x
          f g : ↑↑s → R
          ⊢ Eq (HAdd.hAdd (s.attach.sum fun x => HSMul.hSMul (f x) ↑x) (s.attach.sum fun …
        -/
        rfl
        /-
          🎉 no goals
        -/
      /-
        case refine_1.map_smul'
        R : Type u_1
        M : Type u_2
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        N : Submodule R M
        I : IsNoetherianRing R
        hN : N.FG
        s : Finset M
        hs : Eq (Submodule.span R ↑s) N
        this✝¹ : DecidableEq M
        this✝ : DecidableEq R
        this : ∀ (x : M), Membership.mem s x → Membership.mem N x
        ⊢ ∀ (m : R) (x : ↑↑s → R), Eq ({ toFun := fun f => ⟨s.attach.sum fun i => HSMu …
      -/
    · intro c f
      /-
        case refine_1.map_smul'
        R : Type u_1
        M : Type u_2
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        N : Submodule R M
        I : IsNoetherianRing R
        hN : N.FG
        s : Finset M
        hs : Eq (Submodule.span R ↑s) N
        this✝¹ : DecidableEq M
        this✝ : DecidableEq R
        this : ∀ (x : M), Membership.mem s x → Membership.mem N x
        c : R
        f : ↑↑s → R
        ⊢ Eq ({ toFun := fun f => ⟨s.attach.sum fun i => HSMul.hSMul (f i) ↑i, ⋯⟩, map …
      -/
      apply Subtype.eq
      /-
        case refine_1.map_smul'.a
        R : Type u_1
        M : Type u_2
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        N : Submodule R M
        I : IsNoetherianRing R
        hN : N.FG
        s : Finset M
        hs : Eq (Submodule.span R ↑s) N
        this✝¹ : DecidableEq M
        this✝ : DecidableEq R
        this : ∀ (x : M), Membership.mem s x → Membership.mem N x
        c : R
        f : ↑↑s → R
        ⊢ Eq ↑({ toFun := fun f => ⟨s.attach.sum fun i => HSMul.hSMul (f i) ↑i, ⋯⟩, ma …
      -/
      change (∑ i ∈ s.attach, (c • f i) • _) = _
      /-
        case refine_1.map_smul'.a
        R : Type u_1
        M : Type u_2
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        N : Submodule R M
        I : IsNoetherianRing R
        hN : N.FG
        s : Finset M
        hs : Eq (Submodule.span R ↑s) N
        this✝¹ : DecidableEq M
        this✝ : DecidableEq R
        this : ∀ (x : M), Membership.mem s x → Membership.mem N x
        c : R
        f : ↑↑s → R
        ⊢ Eq (s.attach.sum fun i => HSMul.hSMul (HSMul.hSMul c (f i)) ↑i) ↑(HSMul.hSMu …
      -/
      simp only [smul_eq_mul, mul_smul]
      /-
        case refine_1.map_smul'.a
        R : Type u_1
        M : Type u_2
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        N : Submodule R M
        I : IsNoetherianRing R
        hN : N.FG
        s : Finset M
        hs : Eq (Submodule.span R ↑s) N
        this✝¹ : DecidableEq M
        this✝ : DecidableEq R
        this : ∀ (x : M), Membership.mem s x → Membership.mem N x
        c : R
        f : ↑↑s → R
        ⊢ Eq (s.attach.sum fun x => HSMul.hSMul c (HSMul.hSMul (f x) ↑x)) ↑(HSMul.hSMu …
      -/
      exact Finset.smul_sum.symm
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      N : Submodule R M
      I : IsNoetherianRing R
      hN : N.FG
      s : Finset M
      hs : Eq (Submodule.span R ↑s) N
      this✝¹ : DecidableEq M
      this✝ : DecidableEq R
      this : ∀ (x : M), Membership.mem s x → Membership.mem N x
      ⊢ Eq (LinearMap.range { toFun := fun f => ⟨s.attach.sum fun i => HSMul.hSMul ( …
    -/
  · rw [LinearMap.range_eq_top]
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      N : Submodule R M
      I : IsNoetherianRing R
      hN : N.FG
      s : Finset M
      hs : Eq (Submodule.span R ↑s) N
      this✝¹ : DecidableEq M
      this✝ : DecidableEq R
      this : ∀ (x : M), Membership.mem s x → Membership.mem N x
      ⊢ Function.Surjective ⇑{ toFun := fun f => ⟨s.attach.sum fun i => HSMul.hSMul  …
    -/
    rintro ⟨n, hn⟩
    /-
      case refine_2.mk
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      N : Submodule R M
      I : IsNoetherianRing R
      hN : N.FG
      s : Finset M
      hs : Eq (Submodule.span R ↑s) N
      this✝¹ : DecidableEq M
      this✝ : DecidableEq R
      this : ∀ (x : M), Membership.mem s x → Membership.mem N x
      n : M
      hn : Membership.mem N n
      ⊢ Exists fun a => Eq ({ toFun := fun f => ⟨s.attach.sum fun i => HSMul.hSMul ( …
    -/
    change n ∈ N at hn
    /-
      case refine_2.mk
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      N : Submodule R M
      I : IsNoetherianRing R
      hN : N.FG
      s : Finset M
      hs : Eq (Submodule.span R ↑s) N
      this✝¹ : DecidableEq M
      this✝ : DecidableEq R
      this : ∀ (x : M), Membership.mem s x → Membership.mem N x
      n : M
      hn : Membership.mem N n
      ⊢ Exists fun a => Eq ({ toFun := fun f => ⟨s.attach.sum fun i => HSMul.hSMul ( …
    -/
    rw [← hs, ← Set.image_id (s : Set M), Finsupp.mem_span_image_iff_linearCombination] at hn
    /-
      case refine_2.mk
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      N : Submodule R M
      I : IsNoetherianRing R
      hN : N.FG
      s : Finset M
      hs : Eq (Submodule.span R ↑s) N
      this✝¹ : DecidableEq M
      this✝ : DecidableEq R
      this : ∀ (x : M), Membership.mem s x → Membership.mem N x
      n : M
      hn✝ : Membership.mem N n
      hn : Exists fun l => And (Membership.mem (Finsupp.supported R R ↑s) l) (Eq ((F …
      ⊢ Exists fun a => Eq ({ toFun := fun f => ⟨s.attach.sum fun i => HSMul.hSMul ( …
    -/
    rcases hn with ⟨l, hl1, hl2⟩
    /-
      case refine_2.mk.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      N : Submodule R M
      I : IsNoetherianRing R
      hN : N.FG
      s : Finset M
      hs : Eq (Submodule.span R ↑s) N
      this✝¹ : DecidableEq M
      this✝ : DecidableEq R
      this : ∀ (x : M), Membership.mem s x → Membership.mem N x
      n : M
      hn : Membership.mem N n
      l : Finsupp M R
      hl1 : Membership.mem (Finsupp.supported R R ↑s) l
      hl2 : Eq ((Finsupp.linearCombination R id) l) n
      ⊢ Exists fun a => Eq ({ toFun := fun f => ⟨s.attach.sum fun i => HSMul.hSMul ( …
    -/
    refine ⟨fun x => l x, Subtype.ext ?_⟩
    /-
      case refine_2.mk.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      N : Submodule R M
      I : IsNoetherianRing R
      hN : N.FG
      s : Finset M
      hs : Eq (Submodule.span R ↑s) N
      this✝¹ : DecidableEq M
      this✝ : DecidableEq R
      this : ∀ (x : M), Membership.mem s x → Membership.mem N x
      n : M
      hn : Membership.mem N n
      l : Finsupp M R
      hl1 : Membership.mem (Finsupp.supported R R ↑s) l
      hl2 : Eq ((Finsupp.linearCombination R id) l) n
      ⊢ Eq ↑({ toFun := fun f => ⟨s.attach.sum fun i => HSMul.hSMul (f i) ↑i, ⋯⟩, ma …
    -/
    change (∑ i ∈ s.attach, l i • (i : M)) = n
    rw [s.sum_attach fun i ↦ l i • i, ← hl2,
      Finsupp.linearCombination_apply, Finsupp.sum, eq_comm]
    /-
      case refine_2.mk.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      N : Submodule R M
      I : IsNoetherianRing R
      hN : N.FG
      s : Finset M
      hs : Eq (Submodule.span R ↑s) N
      this✝¹ : DecidableEq M
      this✝ : DecidableEq R
      this : ∀ (x : M), Membership.mem s x → Membership.mem N x
      n : M
      hn : Membership.mem N n
      l : Finsupp M R
      hl1 : Membership.mem (Finsupp.supported R R ↑s) l
      hl2 : Eq ((Finsupp.linearCombination R id) l) n
      ⊢ Eq (l.support.sum fun a => HSMul.hSMul (l a) (id a)) (s.sum fun x => HSMul.h …
    -/
    refine Finset.sum_subset hl1 fun x _ hx => ?_
    /-
      case refine_2.mk.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      N : Submodule R M
      I : IsNoetherianRing R
      hN : N.FG
      s : Finset M
      hs : Eq (Submodule.span R ↑s) N
      this✝¹ : DecidableEq M
      this✝ : DecidableEq R
      this : ∀ (x : M), Membership.mem s x → Membership.mem N x
      n : M
      hn : Membership.mem N n
      l : Finsupp M R
      hl1 : Membership.mem (Finsupp.supported R R ↑s) l
      hl2 : Eq ((Finsupp.linearCombination R id) l) n
      x : M
      x✝ : Membership.mem s x
      hx : Not (Membership.mem l.support x)
      ⊢ Eq (HSMul.hSMul (l x) (id x)) 0
    -/
    rw [Finsupp.not_mem_support_iff.1 hx, zero_smul]
    /-
      🎉 no goals
    -/


instance isNoetherian_of_isNoetherianRing_of_finite (R M : Type*)
    [Ring R] [AddCommGroup M] [Module R M] [IsNoetherianRing R] [Module.Finite R M] :
    IsNoetherian R M :=
  have : IsNoetherian R (⊤ : Submodule R M) :=
    isNoetherian_of_fg_of_noetherian _ <| Module.finite_def.mp inferInstance
  isNoetherian_of_linearEquiv (LinearEquiv.ofTop (⊤ : Submodule R M) rfl)


/-- In a module over a Noetherian ring, the submodule generated by finitely many vectors is
Noetherian. -/
theorem isNoetherian_span_of_finite (R) {M} [Ring R] [AddCommGroup M] [Module R M]
    [IsNoetherianRing R] {A : Set M} (hA : A.Finite) : IsNoetherian R (Submodule.span R A) :=
  isNoetherian_of_fg_of_noetherian _ (Submodule.fg_def.mpr ⟨A, hA, rfl⟩)


theorem isNoetherianRing_of_surjective (R) [Ring R] (S) [Ring S] (f : R →+* S)
    (hf : Function.Surjective f) [H : IsNoetherianRing R] : IsNoetherianRing S :=
  isNoetherian_mk ⟨OrderEmbedding.wellFounded (Ideal.orderEmbeddingOfSurjective f hf).dual H.wf⟩


instance isNoetherianRing_range {R} [Ring R] {S} [Ring S] (f : R →+* S) [IsNoetherianRing R] :
    IsNoetherianRing f.range :=
  isNoetherianRing_of_surjective R f.range f.rangeRestrict f.rangeRestrict_surjective


theorem isNoetherianRing_of_ringEquiv (R) [Ring R] {S} [Ring S] (f : R ≃+* S) [IsNoetherianRing R] :
    IsNoetherianRing S :=
  isNoetherianRing_of_surjective R S f.toRingHom f.toEquiv.surjective

