/-- `N.colon P` is the ideal of all elements `r : R` such that `r • P ⊆ N`. -/
def colon (N P : Submodule R M) : Ideal R where
  carrier := {r : R | (r • P : Set M) ⊆ N}
  add_mem' ha hb :=
                                                                               /-
                                                                                 R : Type u_1
                                                                                 M : Type u_2
                                                                                 M' : Type u_3
                                                                                 F : Type u_4
                                                                                 G : Type u_5
                                                                                 inst✝² : Semiring R
                                                                                 inst✝¹ : AddCommMonoid M
                                                                                 inst✝ : Module R M
                                                                                 N✝ P✝ N P : Submodule R M
                                                                                 a✝ b✝ : R
                                                                                 ha : Membership.mem (setOf fun r => HasSubset.Subset (HSMul.hSMul r ↑P) ↑N) a✝
                                                                                 hb : Membership.mem (setOf fun r => HasSubset.Subset (HSMul.hSMul r ↑P) ↑N) b✝
                                                                                 ⊢ Eq (HAdd.hAdd ↑N ↑N) ↑N
                                                                               -/
    (Set.add_smul_subset _ _ _).trans ((Set.add_subset_add ha hb).trans_eq (by simp [← coe_sup]))
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                  /-
                    R : Type u_1
                    M : Type u_2
                    M' : Type u_3
                    F : Type u_4
                    G : Type u_5
                    inst✝² : Semiring R
                    inst✝¹ : AddCommMonoid M
                    inst✝ : Module R M
                    N✝ P✝ N P : Submodule R M
                    ⊢ Membership.mem { carrier := setOf fun r => HasSubset.Subset (HSMul.hSMul r ↑ …
                  -/
  zero_mem' := by simp [Set.zero_smul_set P.nonempty]
                  /-
                    🎉 no goals
                  -/
  smul_mem' r := by
    /-
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      N✝ P✝ N P : Submodule R M
      r : R
      ⊢ ∀ {x : R}, Membership.mem { carrier := setOf fun r => HasSubset.Subset (HSMu …
    -/
    simp only [Set.mem_setOf_eq, smul_eq_mul, mul_smul, Set.smul_set_subset_iff]
    /-
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      N✝ P✝ N P : Submodule R M
      r : R
      ⊢ ∀ {x : R}, (∀ ⦃b : M⦄, Membership.mem (↑P) b → Membership.mem (↑N) (HSMul.hS …
    -/
    intro x hx y hy
    /-
      R : Type u_1
      M : Type u_2
      M' : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      N✝ P✝ N P : Submodule R M
      r x : R
      hx : ∀ ⦃b : M⦄, Membership.mem (↑P) b → Membership.mem (↑N) (HSMul.hSMul x b)
      y : M
      hy : Membership.mem (↑P) y
      ⊢ Membership.mem (↑N) (HSMul.hSMul r (HSMul.hSMul x y))
    -/
    exact N.smul_mem _ (hx hy)
    /-
      🎉 no goals
    -/


theorem mem_colon {r} : r ∈ N.colon P ↔ ∀ p ∈ P, r • p ∈ N := Set.smul_set_subset_iff


theorem mem_colon' {r} : r ∈ N.colon P ↔ P ≤ comap (r • (LinearMap.id : M →ₗ[R] M)) N :=
  mem_colon


@[simp]
theorem colon_top {I : Ideal R} : I.colon ⊤ = I := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    I : Ideal R
    ⊢ Eq (Submodule.colon I Top.top) I
  -/
  simp_rw [SetLike.ext_iff, mem_colon, smul_eq_mul]
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    I : Ideal R
    ⊢ ∀ (x : R), Iff (∀ (p : R), Membership.mem Top.top p → Membership.mem I (HMul …
  -/
  exact fun x ↦ ⟨fun h ↦ mul_one x ▸ h 1 trivial, fun h _ _ ↦ I.mul_mem_right _ h⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem colon_bot : colon ⊥ N = N.annihilator := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    N : Submodule R M
    ⊢ Eq (Bot.bot.colon N) N.annihilator
  -/
  simp_rw [SetLike.ext_iff, mem_colon, mem_annihilator, mem_bot, forall_const]
  /-
    🎉 no goals
  -/


theorem colon_mono (hn : N₁ ≤ N₂) (hp : P₁ ≤ P₂) : N₁.colon P₂ ≤ N₂.colon P₁ := fun _ hrnp =>
  mem_colon.2 fun p₁ hp₁ => hn <| mem_colon.1 hrnp p₁ <| hp hp₁


theorem iInf_colon_iSup (ι₁ : Sort*) (f : ι₁ → Submodule R M) (ι₂ : Sort*)
    (g : ι₂ → Submodule R M) : (⨅ i, f i).colon (⨆ j, g j) = ⨅ (i) (j), (f i).colon (g j) :=
  le_antisymm (le_iInf fun _ => le_iInf fun _ => colon_mono (iInf_le _ _) (le_iSup _ _)) fun _ H =>
    mem_colon'.2 <|
      iSup_le fun j =>
        map_le_iff_le_comap.1 <|
          le_iInf fun i =>
            map_le_iff_le_comap.2 <|
              mem_colon'.1 <|
                have := (mem_iInf _).1 H i
                have := (mem_iInf _).1 this j
                this


@[simp]
theorem mem_colon_singleton {N : Submodule R M} {x : M} {r : R} :
    r ∈ N.colon (Submodule.span R {x}) ↔ r • x ∈ N :=
  calc
    r ∈ N.colon (Submodule.span R {x}) ↔ ∀ a : R, r • a • x ∈ N := by
      /-
        R : Type u_1
        M : Type u_2
        inst✝² : CommSemiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        N : Submodule R M
        x : M
        r : R
        ⊢ Iff (Membership.mem (N.colon (Submodule.span R (Singleton.singleton x))) r)  …
      -/
      simp [Submodule.mem_colon, Submodule.mem_span_singleton]
      /-
        🎉 no goals
      -/
                        /-
                          R : Type u_1
                          M : Type u_2
                          inst✝² : CommSemiring R
                          inst✝¹ : AddCommMonoid M
                          inst✝ : Module R M
                          N : Submodule R M
                          x : M
                          r : R
                          ⊢ Iff (∀ (a : R), Membership.mem N (HSMul.hSMul r (HSMul.hSMul a x))) (Members …
                        -/
    _ ↔ r • x ∈ N := by simp_rw [fun (a : R) ↦ smul_comm r a x]; exact SetLike.forall_smul_mem_iff
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp]
theorem _root_.Ideal.mem_colon_singleton {I : Ideal R} {x r : R} :
    r ∈ I.colon (Ideal.span {x}) ↔ r * x ∈ I := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    I : Ideal R
    x r : R
    ⊢ Iff (Membership.mem (Submodule.colon I (Ideal.span (Singleton.singleton x))) …
  -/
  simp only [← Ideal.submodule_span_eq, Submodule.mem_colon_singleton, smul_eq_mul]
  /-
    🎉 no goals
  -/


@[simp]
lemma annihilator_map_mkQ_eq_colon : annihilator (P.map N.mkQ) = N.colon P := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    N P : Submodule R M
    ⊢ Eq (Submodule.map N.mkQ P).annihilator (N.colon P)
  -/
  ext
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    N P : Submodule R M
    x✝ : R
    ⊢ Iff (Membership.mem (Submodule.map N.mkQ P).annihilator x✝) (Membership.mem  …
  -/
  rw [mem_annihilator, mem_colon]
  exact ⟨fun H p hp ↦ (Quotient.mk_eq_zero N).1 (H (Quotient.mk p) (mem_map_of_mem hp)),
    fun H _ ⟨p, hp, hpm⟩ ↦ hpm ▸ ((Quotient.mk_eq_zero N).2 <| H p hp)⟩


theorem annihilator_quotient {N : Submodule R M} :
    Module.annihilator R (M ⧸ N) = N.colon ⊤ := by
  simp_rw [SetLike.ext_iff, Module.mem_annihilator, ←annihilator_map_mkQ_eq_colon, mem_annihilator,
      map_top, LinearMap.range_eq_top.mpr (mkQ_surjective N), mem_top, forall_true_left,
      forall_const]


theorem _root_.Ideal.annihilator_quotient {I : Ideal R} : Module.annihilator R (R ⧸ I) = I := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    ⊢ Eq (Module.annihilator R (HasQuotient.Quotient R I)) I
  -/
  rw [Submodule.annihilator_quotient, colon_top]
  /-
    🎉 no goals
  -/


