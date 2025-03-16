/-- An `I`-filtration on the module `M` is a sequence of decreasing submodules `N i` such that
`I • (N i) ≤ N (i + 1)`. Note that we do not require the filtration to start from `⊤`. -/
@[ext]
structure Ideal.Filtration (M : Type*) [AddCommGroup M] [Module R M] where
  N : ℕ → Submodule R M
  mono : ∀ i, N (i + 1) ≤ N i
  smul_le : ∀ i, I • N i ≤ N (i + 1)


theorem pow_smul_le (i j : ℕ) : I ^ i • F.N j ≤ F.N (i + j) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    i j : Nat
    ⊢ LE.le (HSMul.hSMul (HPow.hPow I i) (F.N j)) (F.N (HAdd.hAdd i j))
  -/
  induction' i with _ ih
    /-
      case zero
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      j : Nat
      ⊢ LE.le (HSMul.hSMul (HPow.hPow I 0) (F.N j)) (F.N (HAdd.hAdd 0 j))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      j n✝ : Nat
      ih : LE.le (HSMul.hSMul (HPow.hPow I n✝) (F.N j)) (F.N (HAdd.hAdd n✝ j))
      ⊢ LE.le (HSMul.hSMul (HPow.hPow I (HAdd.hAdd n✝ 1)) (F.N j)) (F.N (HAdd.hAdd ( …
    -/
  · rw [pow_succ', mul_smul, add_assoc, add_comm 1, ← add_assoc]
    /-
      case succ
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      j n✝ : Nat
      ih : LE.le (HSMul.hSMul (HPow.hPow I n✝) (F.N j)) (F.N (HAdd.hAdd n✝ j))
      ⊢ LE.le (HSMul.hSMul I (HSMul.hSMul (HPow.hPow I n✝) (F.N j))) (F.N (HAdd.hAdd …
    -/
    exact (smul_mono_right _ ih).trans (F.smul_le _)
    /-
      🎉 no goals
    -/


theorem pow_smul_le_pow_smul (i j k : ℕ) : I ^ (i + k) • F.N j ≤ I ^ k • F.N (i + j) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    i j k : Nat
    ⊢ LE.le (HSMul.hSMul (HPow.hPow I (HAdd.hAdd i k)) (F.N j)) (HSMul.hSMul (HPow …
  -/
  rw [add_comm, pow_add, mul_smul]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    i j k : Nat
    ⊢ LE.le (HSMul.hSMul (HPow.hPow I k) (HSMul.hSMul (HPow.hPow I i) (F.N j))) (H …
  -/
  exact smul_mono_right _ (F.pow_smul_le i j)
  /-
    🎉 no goals
  -/


protected theorem antitone : Antitone F.N :=
  antitone_nat_of_succ_le F.mono


/-- The trivial `I`-filtration of `N`. -/
@[simps]
def _root_.Ideal.trivialFiltration (I : Ideal R) (N : Submodule R M) : I.Filtration M where
  N _ := N
  mono _ := le_rfl
  smul_le _ := Submodule.smul_le_right


/-- The `sup` of two `I.Filtration`s is an `I.Filtration`. -/
instance : Max (I.Filtration M) :=
  ⟨fun F F' =>
    ⟨F.N ⊔ F'.N, fun i => sup_le_sup (F.mono i) (F'.mono i), fun i =>
      (Submodule.smul_sup _ _ _).trans_le <| sup_le_sup (F.smul_le i) (F'.smul_le i)⟩⟩


/-- The `sSup` of a family of `I.Filtration`s is an `I.Filtration`. -/
instance : SupSet (I.Filtration M) :=
  ⟨fun S =>
    { N := sSup (Ideal.Filtration.N '' S)
      mono := fun i => by
        /-
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          I : Ideal R
          F F' : I.Filtration M
          S : Set (I.Filtration M)
          i : Nat
          ⊢ LE.le (SupSet.sSup (Set.image Ideal.Filtration.N S) (HAdd.hAdd i 1)) (SupSet …
        -/
        apply sSup_le_sSup_of_forall_exists_le _
        /-
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          I : Ideal R
          F F' : I.Filtration M
          S : Set (I.Filtration M)
          i : Nat
          ⊢ ∀ (x : Submodule R M), Membership.mem (Set.range fun f => ↑f (HAdd.hAdd i 1) …
        -/
        rintro _ ⟨⟨_, F, hF, rfl⟩, rfl⟩
        /-
          case intro.mk.intro.intro
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          I : Ideal R
          F✝ F' : I.Filtration M
          S : Set (I.Filtration M)
          i : Nat
          F : I.Filtration M
          hF : Membership.mem S F
          ⊢ Exists fun y => And (Membership.mem (Set.range fun f => ↑f i) y) (LE.le ((fu …
        -/
        exact ⟨_, ⟨⟨_, F, hF, rfl⟩, rfl⟩, F.mono i⟩
        /-
          🎉 no goals
        -/
      smul_le := fun i => by
        /-
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          I : Ideal R
          F F' : I.Filtration M
          S : Set (I.Filtration M)
          i : Nat
          ⊢ LE.le (HSMul.hSMul I (SupSet.sSup (Set.image Ideal.Filtration.N S) i)) (SupS …
        -/
        rw [sSup_eq_iSup', iSup_apply, Submodule.smul_iSup, iSup_apply]
        /-
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          I : Ideal R
          F F' : I.Filtration M
          S : Set (I.Filtration M)
          i : Nat
          ⊢ LE.le (iSup fun i_1 => HSMul.hSMul I (↑i_1 i)) (iSup fun i_1 => ↑i_1 (HAdd.h …
        -/
        apply iSup_mono _
        /-
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          I : Ideal R
          F F' : I.Filtration M
          S : Set (I.Filtration M)
          i : Nat
          ⊢ ∀ (i_1 : ↑(Set.image Ideal.Filtration.N S)), LE.le (HSMul.hSMul I (↑i_1 i))  …
        -/
        rintro ⟨_, F, hF, rfl⟩
        /-
          case mk.intro.intro
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          I : Ideal R
          F✝ F' : I.Filtration M
          S : Set (I.Filtration M)
          i : Nat
          F : I.Filtration M
          hF : Membership.mem S F
          ⊢ LE.le (HSMul.hSMul I (↑⟨F.N, ⋯⟩ i)) (↑⟨F.N, ⋯⟩ (HAdd.hAdd i 1))
        -/
        exact F.smul_le i }⟩
        /-
          🎉 no goals
        -/


/-- The `inf` of two `I.Filtration`s is an `I.Filtration`. -/
instance : Min (I.Filtration M) :=
  ⟨fun F F' =>
    ⟨F.N ⊓ F'.N, fun i => inf_le_inf (F.mono i) (F'.mono i), fun i =>
      (smul_inf_le _ _ _).trans <| inf_le_inf (F.smul_le i) (F'.smul_le i)⟩⟩


/-- The `sInf` of a family of `I.Filtration`s is an `I.Filtration`. -/
instance : InfSet (I.Filtration M) :=
  ⟨fun S =>
    { N := sInf (Ideal.Filtration.N '' S)
      mono := fun i => by
        /-
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          I : Ideal R
          F F' : I.Filtration M
          S : Set (I.Filtration M)
          i : Nat
          ⊢ LE.le (InfSet.sInf (Set.image Ideal.Filtration.N S) (HAdd.hAdd i 1)) (InfSet …
        -/
        apply sInf_le_sInf_of_forall_exists_le _
        /-
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          I : Ideal R
          F F' : I.Filtration M
          S : Set (I.Filtration M)
          i : Nat
          ⊢ ∀ (x : Submodule R M), Membership.mem (Set.range fun f => ↑f i) x → Exists f …
        -/
        rintro _ ⟨⟨_, F, hF, rfl⟩, rfl⟩
        /-
          case intro.mk.intro.intro
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          I : Ideal R
          F✝ F' : I.Filtration M
          S : Set (I.Filtration M)
          i : Nat
          F : I.Filtration M
          hF : Membership.mem S F
          ⊢ Exists fun y => And (Membership.mem (Set.range fun f => ↑f (HAdd.hAdd i 1))  …
        -/
        exact ⟨_, ⟨⟨_, F, hF, rfl⟩, rfl⟩, F.mono i⟩
        /-
          🎉 no goals
        -/
      smul_le := fun i => by
        /-
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          I : Ideal R
          F F' : I.Filtration M
          S : Set (I.Filtration M)
          i : Nat
          ⊢ LE.le (HSMul.hSMul I (InfSet.sInf (Set.image Ideal.Filtration.N S) i)) (InfS …
        -/
        rw [sInf_eq_iInf', iInf_apply, iInf_apply]
        /-
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          I : Ideal R
          F F' : I.Filtration M
          S : Set (I.Filtration M)
          i : Nat
          ⊢ LE.le (HSMul.hSMul I (iInf fun i_1 => ↑i_1 i)) (iInf fun i_1 => ↑i_1 (HAdd.h …
        -/
        refine smul_iInf_le.trans ?_
        /-
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          I : Ideal R
          F F' : I.Filtration M
          S : Set (I.Filtration M)
          i : Nat
          ⊢ LE.le (iInf fun i_1 => HSMul.hSMul I (↑i_1 i)) (iInf fun i_1 => ↑i_1 (HAdd.h …
        -/
        apply iInf_mono _
        /-
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          I : Ideal R
          F F' : I.Filtration M
          S : Set (I.Filtration M)
          i : Nat
          ⊢ ∀ (i_1 : ↑(Set.image Ideal.Filtration.N S)), LE.le (HSMul.hSMul I (↑i_1 i))  …
        -/
        rintro ⟨_, F, hF, rfl⟩
        /-
          case mk.intro.intro
          R : Type u_1
          M : Type u_2
          inst✝² : CommRing R
          inst✝¹ : AddCommGroup M
          inst✝ : Module R M
          I : Ideal R
          F✝ F' : I.Filtration M
          S : Set (I.Filtration M)
          i : Nat
          F : I.Filtration M
          hF : Membership.mem S F
          ⊢ LE.le (HSMul.hSMul I (↑⟨F.N, ⋯⟩ i)) (↑⟨F.N, ⋯⟩ (HAdd.hAdd i 1))
        -/
        exact F.smul_le i }⟩
        /-
          🎉 no goals
        -/


instance : Top (I.Filtration M) :=
  ⟨I.trivialFiltration ⊤⟩


instance : Bot (I.Filtration M) :=
  ⟨I.trivialFiltration ⊥⟩


@[simp]
theorem sup_N : (F ⊔ F').N = F.N ⊔ F'.N :=
  rfl


@[simp]
theorem sSup_N (S : Set (I.Filtration M)) : (sSup S).N = sSup (Ideal.Filtration.N '' S) :=
  rfl


@[simp]
theorem inf_N : (F ⊓ F').N = F.N ⊓ F'.N :=
  rfl


@[simp]
theorem sInf_N (S : Set (I.Filtration M)) : (sInf S).N = sInf (Ideal.Filtration.N '' S) :=
  rfl


@[simp]
theorem top_N : (⊤ : I.Filtration M).N = ⊤ :=
  rfl


@[simp]
theorem bot_N : (⊥ : I.Filtration M).N = ⊥ :=
  rfl


@[simp]
theorem iSup_N {ι : Sort*} (f : ι → I.Filtration M) : (iSup f).N = ⨆ i, (f i).N :=
  congr_arg sSup (Set.range_comp _ _).symm


@[simp]
theorem iInf_N {ι : Sort*} (f : ι → I.Filtration M) : (iInf f).N = ⨅ i, (f i).N :=
  congr_arg sInf (Set.range_comp _ _).symm


instance : CompleteLattice (I.Filtration M) :=
  Function.Injective.completeLattice Ideal.Filtration.N
    (fun _ _ => Ideal.Filtration.ext) sup_N inf_N
    (fun _ => sSup_image) (fun _ => sInf_image) top_N bot_N


instance : Inhabited (I.Filtration M) :=
  ⟨⊥⟩


/-- An `I` filtration is stable if `I • F.N n = F.N (n+1)` for large enough `n`. -/
def Stable : Prop :=
  ∃ n₀, ∀ n ≥ n₀, I • F.N n = F.N (n + 1)


/-- The trivial stable `I`-filtration of `N`. -/
@[simps]
def _root_.Ideal.stableFiltration (I : Ideal R) (N : Submodule R M) : I.Filtration M where
  N i := I ^ i • N
               /-
                 R : Type u_1
                 M : Type u_2
                 inst✝² : CommRing R
                 inst✝¹ : AddCommGroup M
                 inst✝ : Module R M
                 I✝ : Ideal R
                 F F' : I✝.Filtration M
                 I : Ideal R
                 N : Submodule R M
                 i : Nat
                 ⊢ LE.le ((fun i => HSMul.hSMul (HPow.hPow I i) N) (HAdd.hAdd i 1)) ((fun i =>  …
               -/
  mono i := by dsimp only; rw [add_comm, pow_add, mul_smul]; exact Submodule.smul_le_right
                                                             /-
                                                               🎉 no goals
                                                             -/
                  /-
                    R : Type u_1
                    M : Type u_2
                    inst✝² : CommRing R
                    inst✝¹ : AddCommGroup M
                    inst✝ : Module R M
                    I✝ : Ideal R
                    F F' : I✝.Filtration M
                    I : Ideal R
                    N : Submodule R M
                    i : Nat
                    ⊢ LE.le (HSMul.hSMul I ((fun i => HSMul.hSMul (HPow.hPow I i) N) i)) ((fun i = …
                  -/
  smul_le i := by dsimp only; rw [add_comm, pow_add, mul_smul, pow_one]
                              /-
                                🎉 no goals
                              -/


theorem _root_.Ideal.stableFiltration_stable (I : Ideal R) (N : Submodule R M) :
    (I.stableFiltration N).Stable := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    ⊢ (I.stableFiltration N).Stable
  -/
  use 0
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    ⊢ ∀ (n : Nat), GE.ge n 0 → Eq (HSMul.hSMul I ((I.stableFiltration N).N n)) ((I …
  -/
  intro n _
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    n : Nat
    a✝ : GE.ge n 0
    ⊢ Eq (HSMul.hSMul I ((I.stableFiltration N).N n)) ((I.stableFiltration N).N (H …
  -/
  dsimp
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    N : Submodule R M
    n : Nat
    a✝ : GE.ge n 0
    ⊢ Eq (HSMul.hSMul I (HSMul.hSMul (HPow.hPow I n) N)) (HSMul.hSMul (HPow.hPow I …
  -/
  rw [add_comm, pow_add, mul_smul, pow_one]
  /-
    🎉 no goals
  -/


theorem Stable.exists_pow_smul_eq (h : F.Stable) : ∃ n₀, ∀ k, F.N (n₀ + k) = I ^ k • F.N n₀ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    h : F.Stable
    ⊢ Exists fun n₀ => ∀ (k : Nat), Eq (F.N (HAdd.hAdd n₀ k)) (HSMul.hSMul (HPow.h …
  -/
  obtain ⟨n₀, hn⟩ := h
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    n₀ : Nat
    hn : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
    ⊢ Exists fun n₀ => ∀ (k : Nat), Eq (F.N (HAdd.hAdd n₀ k)) (HSMul.hSMul (HPow.h …
  -/
  use n₀
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    n₀ : Nat
    hn : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
    ⊢ ∀ (k : Nat), Eq (F.N (HAdd.hAdd n₀ k)) (HSMul.hSMul (HPow.hPow I k) (F.N n₀))
  -/
  intro k
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    n₀ : Nat
    hn : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
    k : Nat
    ⊢ Eq (F.N (HAdd.hAdd n₀ k)) (HSMul.hSMul (HPow.hPow I k) (F.N n₀))
  -/
  induction' k with _ ih
    /-
      case h.zero
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      hn : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
      ⊢ Eq (F.N (HAdd.hAdd n₀ 0)) (HSMul.hSMul (HPow.hPow I 0) (F.N n₀))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.succ
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      hn : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
      n✝ : Nat
      ih : Eq (F.N (HAdd.hAdd n₀ n✝)) (HSMul.hSMul (HPow.hPow I n✝) (F.N n₀))
      ⊢ Eq (F.N (HAdd.hAdd n₀ (HAdd.hAdd n✝ 1))) (HSMul.hSMul (HPow.hPow I (HAdd.hAd …
    -/
  · rw [← add_assoc, ← hn, ih, add_comm, pow_add, mul_smul, pow_one]
    /-
      case h.succ.a
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      hn : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
      n✝ : Nat
      ih : Eq (F.N (HAdd.hAdd n₀ n✝)) (HSMul.hSMul (HPow.hPow I n✝) (F.N n₀))
      ⊢ GE.ge (HAdd.hAdd n₀ n✝) n₀
    -/
    omega
    /-
      🎉 no goals
    -/


theorem Stable.exists_pow_smul_eq_of_ge (h : F.Stable) :
    ∃ n₀, ∀ n ≥ n₀, F.N n = I ^ (n - n₀) • F.N n₀ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    h : F.Stable
    ⊢ Exists fun n₀ => ∀ (n : Nat), GE.ge n n₀ → Eq (F.N n) (HSMul.hSMul (HPow.hPo …
  -/
  obtain ⟨n₀, hn₀⟩ := h.exists_pow_smul_eq
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    h : F.Stable
    n₀ : Nat
    hn₀ : ∀ (k : Nat), Eq (F.N (HAdd.hAdd n₀ k)) (HSMul.hSMul (HPow.hPow I k) (F.N …
    ⊢ Exists fun n₀ => ∀ (n : Nat), GE.ge n n₀ → Eq (F.N n) (HSMul.hSMul (HPow.hPo …
  -/
  use n₀
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    h : F.Stable
    n₀ : Nat
    hn₀ : ∀ (k : Nat), Eq (F.N (HAdd.hAdd n₀ k)) (HSMul.hSMul (HPow.hPow I k) (F.N …
    ⊢ ∀ (n : Nat), GE.ge n n₀ → Eq (F.N n) (HSMul.hSMul (HPow.hPow I (HSub.hSub n  …
  -/
  intro n hn
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    h : F.Stable
    n₀ : Nat
    hn₀ : ∀ (k : Nat), Eq (F.N (HAdd.hAdd n₀ k)) (HSMul.hSMul (HPow.hPow I k) (F.N …
    n : Nat
    hn : GE.ge n n₀
    ⊢ Eq (F.N n) (HSMul.hSMul (HPow.hPow I (HSub.hSub n n₀)) (F.N n₀))
  -/
  convert hn₀ (n - n₀)
  /-
    case h.e'_2.h.e'_8
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    h : F.Stable
    n₀ : Nat
    hn₀ : ∀ (k : Nat), Eq (F.N (HAdd.hAdd n₀ k)) (HSMul.hSMul (HPow.hPow I k) (F.N …
    n : Nat
    hn : GE.ge n n₀
    ⊢ Eq n (HAdd.hAdd n₀ (HSub.hSub n n₀))
  -/
  rw [add_comm, tsub_add_cancel_of_le hn]
  /-
    🎉 no goals
  -/


theorem stable_iff_exists_pow_smul_eq_of_ge :
    F.Stable ↔ ∃ n₀, ∀ n ≥ n₀, F.N n = I ^ (n - n₀) • F.N n₀ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    ⊢ Iff F.Stable (Exists fun n₀ => ∀ (n : Nat), GE.ge n n₀ → Eq (F.N n) (HSMul.h …
  -/
  refine ⟨Stable.exists_pow_smul_eq_of_ge, fun h => ⟨h.choose, fun n hn => ?_⟩⟩
  rw [h.choose_spec n hn, h.choose_spec (n + 1) (by omega), smul_smul, ← pow_succ',
    tsub_add_eq_add_tsub hn]


theorem Stable.exists_forall_le (h : F.Stable) (e : F.N 0 ≤ F'.N 0) :
    ∃ n₀, ∀ n, F.N (n + n₀) ≤ F'.N n := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F F' : I.Filtration M
    h : F.Stable
    e : LE.le (F.N 0) (F'.N 0)
    ⊢ Exists fun n₀ => ∀ (n : Nat), LE.le (F.N (HAdd.hAdd n n₀)) (F'.N n)
  -/
  obtain ⟨n₀, hF⟩ := h
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F F' : I.Filtration M
    e : LE.le (F.N 0) (F'.N 0)
    n₀ : Nat
    hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
    ⊢ Exists fun n₀ => ∀ (n : Nat), LE.le (F.N (HAdd.hAdd n n₀)) (F'.N n)
  -/
  use n₀
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F F' : I.Filtration M
    e : LE.le (F.N 0) (F'.N 0)
    n₀ : Nat
    hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
    ⊢ ∀ (n : Nat), LE.le (F.N (HAdd.hAdd n n₀)) (F'.N n)
  -/
  intro n
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F F' : I.Filtration M
    e : LE.le (F.N 0) (F'.N 0)
    n₀ : Nat
    hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
    n : Nat
    ⊢ LE.le (F.N (HAdd.hAdd n n₀)) (F'.N n)
  -/
  induction' n with n hn
    /-
      case h.zero
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F F' : I.Filtration M
      e : LE.le (F.N 0) (F'.N 0)
      n₀ : Nat
      hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
      ⊢ LE.le (F.N (HAdd.hAdd 0 n₀)) (F'.N 0)
    -/
  · refine (F.antitone ?_).trans e; simp
                                    /-
                                      🎉 no goals
                                    -/
    /-
      case h.succ
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F F' : I.Filtration M
      e : LE.le (F.N 0) (F'.N 0)
      n₀ : Nat
      hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
      n : Nat
      hn : LE.le (F.N (HAdd.hAdd n n₀)) (F'.N n)
      ⊢ LE.le (F.N (HAdd.hAdd (HAdd.hAdd n 1) n₀)) (F'.N (HAdd.hAdd n 1))
    -/
  · rw [add_right_comm, ← hF]
      /-
        case h.succ
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        I : Ideal R
        F F' : I.Filtration M
        e : LE.le (F.N 0) (F'.N 0)
        n₀ : Nat
        hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
        n : Nat
        hn : LE.le (F.N (HAdd.hAdd n n₀)) (F'.N n)
        ⊢ LE.le (HSMul.hSMul I (F.N (HAdd.hAdd n n₀))) (F'.N (HAdd.hAdd n 1))
      -/
    · exact (smul_mono_right _ hn).trans (F'.smul_le _)
      /-
        🎉 no goals
      -/
    /-
      case h.succ.a
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F F' : I.Filtration M
      e : LE.le (F.N 0) (F'.N 0)
      n₀ : Nat
      hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
      n : Nat
      hn : LE.le (F.N (HAdd.hAdd n n₀)) (F'.N n)
      ⊢ GE.ge (HAdd.hAdd n n₀) n₀
    -/
    simp
    /-
      🎉 no goals
    -/


theorem Stable.bounded_difference (h : F.Stable) (h' : F'.Stable) (e : F.N 0 = F'.N 0) :
    ∃ n₀, ∀ n, F.N (n + n₀) ≤ F'.N n ∧ F'.N (n + n₀) ≤ F.N n := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F F' : I.Filtration M
    h : F.Stable
    h' : F'.Stable
    e : Eq (F.N 0) (F'.N 0)
    ⊢ Exists fun n₀ => ∀ (n : Nat), And (LE.le (F.N (HAdd.hAdd n n₀)) (F'.N n)) (L …
  -/
  obtain ⟨n₁, h₁⟩ := h.exists_forall_le (le_of_eq e)
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F F' : I.Filtration M
    h : F.Stable
    h' : F'.Stable
    e : Eq (F.N 0) (F'.N 0)
    n₁ : Nat
    h₁ : ∀ (n : Nat), LE.le (F.N (HAdd.hAdd n n₁)) (F'.N n)
    ⊢ Exists fun n₀ => ∀ (n : Nat), And (LE.le (F.N (HAdd.hAdd n n₀)) (F'.N n)) (L …
  -/
  obtain ⟨n₂, h₂⟩ := h'.exists_forall_le (le_of_eq e.symm)
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F F' : I.Filtration M
    h : F.Stable
    h' : F'.Stable
    e : Eq (F.N 0) (F'.N 0)
    n₁ : Nat
    h₁ : ∀ (n : Nat), LE.le (F.N (HAdd.hAdd n n₁)) (F'.N n)
    n₂ : Nat
    h₂ : ∀ (n : Nat), LE.le (F'.N (HAdd.hAdd n n₂)) (F.N n)
    ⊢ Exists fun n₀ => ∀ (n : Nat), And (LE.le (F.N (HAdd.hAdd n n₀)) (F'.N n)) (L …
  -/
  use max n₁ n₂
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F F' : I.Filtration M
    h : F.Stable
    h' : F'.Stable
    e : Eq (F.N 0) (F'.N 0)
    n₁ : Nat
    h₁ : ∀ (n : Nat), LE.le (F.N (HAdd.hAdd n n₁)) (F'.N n)
    n₂ : Nat
    h₂ : ∀ (n : Nat), LE.le (F'.N (HAdd.hAdd n n₂)) (F.N n)
    ⊢ ∀ (n : Nat), And (LE.le (F.N (HAdd.hAdd n (Max.max n₁ n₂))) (F'.N n)) (LE.le …
  -/
  intro n
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F F' : I.Filtration M
    h : F.Stable
    h' : F'.Stable
    e : Eq (F.N 0) (F'.N 0)
    n₁ : Nat
    h₁ : ∀ (n : Nat), LE.le (F.N (HAdd.hAdd n n₁)) (F'.N n)
    n₂ : Nat
    h₂ : ∀ (n : Nat), LE.le (F'.N (HAdd.hAdd n n₂)) (F.N n)
    n : Nat
    ⊢ And (LE.le (F.N (HAdd.hAdd n (Max.max n₁ n₂))) (F'.N n)) (LE.le (F'.N (HAdd. …
  -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  refine ⟨(F.antitone ?_).trans (h₁ n), (F'.antitone ?_).trans (h₂ n)⟩ <;> simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- The `R[IX]`-submodule of `M[X]` associated with an `I`-filtration. -/
protected def submodule : Submodule (reesAlgebra I) (PolynomialModule R M) where
  carrier := { f | ∀ i, f i ∈ F.N i }
  add_mem' hf hg i := Submodule.add_mem _ (hf i) (hg i)
  zero_mem' _ := Submodule.zero_mem _
  smul_mem' r f hf i := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F F' : I.Filtration M
      r : Subtype fun x => Membership.mem (reesAlgebra I) x
      f : PolynomialModule R M
      hf : Membership.mem { carrier := setOf fun f => ∀ (i : Nat), Membership.mem (F …
      i : Nat
      ⊢ Membership.mem (F.N i) ((HSMul.hSMul r f) i)
    -/
    rw [Subalgebra.smul_def, PolynomialModule.smul_apply]
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F F' : I.Filtration M
      r : Subtype fun x => Membership.mem (reesAlgebra I) x
      f : PolynomialModule R M
      hf : Membership.mem { carrier := setOf fun f => ∀ (i : Nat), Membership.mem (F …
      i : Nat
      ⊢ Membership.mem (F.N i) ((Finset.HasAntidiagonal.antidiagonal i).sum fun x => …
    -/
    apply Submodule.sum_mem
    /-
      case a
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F F' : I.Filtration M
      r : Subtype fun x => Membership.mem (reesAlgebra I) x
      f : PolynomialModule R M
      hf : Membership.mem { carrier := setOf fun f => ∀ (i : Nat), Membership.mem (F …
      i : Nat
      ⊢ ∀ (c : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal i) …
    -/
    rintro ⟨j, k⟩ e
    /-
      case a.mk
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F F' : I.Filtration M
      r : Subtype fun x => Membership.mem (reesAlgebra I) x
      f : PolynomialModule R M
      hf : Membership.mem { carrier := setOf fun f => ∀ (i : Nat), Membership.mem (F …
      i j k : Nat
      e : Membership.mem (Finset.HasAntidiagonal.antidiagonal i) { fst := j, snd :=  …
      ⊢ Membership.mem (F.N i) (HSMul.hSMul ((↑r).coeff { fst := j, snd := k }.1) (f …
    -/
    rw [Finset.mem_antidiagonal] at e
    /-
      case a.mk
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F F' : I.Filtration M
      r : Subtype fun x => Membership.mem (reesAlgebra I) x
      f : PolynomialModule R M
      hf : Membership.mem { carrier := setOf fun f => ∀ (i : Nat), Membership.mem (F …
      i j k : Nat
      e : Eq (HAdd.hAdd { fst := j, snd := k }.1 { fst := j, snd := k }.2) i
      ⊢ Membership.mem (F.N i) (HSMul.hSMul ((↑r).coeff { fst := j, snd := k }.1) (f …
    -/
    subst e
    /-
      case a.mk
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F F' : I.Filtration M
      r : Subtype fun x => Membership.mem (reesAlgebra I) x
      f : PolynomialModule R M
      hf : Membership.mem { carrier := setOf fun f => ∀ (i : Nat), Membership.mem (F …
      j k : Nat
      ⊢ Membership.mem (F.N (HAdd.hAdd { fst := j, snd := k }.1 { fst := j, snd := k …
    -/
    exact F.pow_smul_le j k (Submodule.smul_mem_smul (r.2 j) (hf k))
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_submodule (f : PolynomialModule R M) : f ∈ F.submodule ↔ ∀ i, f i ∈ F.N i :=
  Iff.rfl


theorem inf_submodule : (F ⊓ F').submodule = F.submodule ⊓ F'.submodule := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F F' : I.Filtration M
    ⊢ Eq (Min.min F F').submodule (Min.min F.submodule F'.submodule)
  -/
  ext
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F F' : I.Filtration M
    x✝ : PolynomialModule R M
    ⊢ Iff (Membership.mem (Min.min F F').submodule x✝) (Membership.mem (Min.min F. …
  -/
  exact forall_and
  /-
    🎉 no goals
  -/


/-- `Ideal.Filtration.submodule` as an `InfHom`. -/
def submoduleInfHom :
    InfHom (I.Filtration M) (Submodule (reesAlgebra I) (PolynomialModule R M)) where
  toFun := Ideal.Filtration.submodule
  map_inf' := inf_submodule


theorem submodule_closure_single :
    AddSubmonoid.closure (⋃ i, single R i '' (F.N i : Set M)) = F.submodule.toAddSubmonoid := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    ⊢ Eq (AddSubmonoid.closure (Set.iUnion fun i => Set.image ⇑(PolynomialModule.s …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      ⊢ LE.le (AddSubmonoid.closure (Set.iUnion fun i => Set.image ⇑(PolynomialModul …
    -/
  · rw [AddSubmonoid.closure_le, Set.iUnion_subset_iff]
    /-
      case a
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      ⊢ ∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑(F. …
    -/
    rintro i _ ⟨m, hm, rfl⟩ j
    /-
      case a.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      i : Nat
      m : M
      hm : Membership.mem (↑(F.N i)) m
      j : Nat
      ⊢ Membership.mem (F.N j) (((PolynomialModule.single R i) m) j)
    -/
    rw [single_apply]
    /-
      case a.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      i : Nat
      m : M
      hm : Membership.mem (↑(F.N i)) m
      j : Nat
      ⊢ Membership.mem (F.N j) (ite (Eq i j) m 0)
    -/
    split_ifs with h
      /-
        case pos
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        I : Ideal R
        F : I.Filtration M
        i : Nat
        m : M
        hm : Membership.mem (↑(F.N i)) m
        j : Nat
        h : Eq i j
        ⊢ Membership.mem (F.N j) m
      -/
    · rwa [← h]
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        I : Ideal R
        F : I.Filtration M
        i : Nat
        m : M
        hm : Membership.mem (↑(F.N i)) m
        j : Nat
        h : Not (Eq i j)
        ⊢ Membership.mem (F.N j) 0
      -/
    · exact (F.N j).zero_mem
      /-
        🎉 no goals
      -/
    /-
      case a
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      ⊢ LE.le F.submodule.toAddSubmonoid (AddSubmonoid.closure (Set.iUnion fun i =>  …
    -/
  · intro f hf
    /-
      case a
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      f : PolynomialModule R M
      hf : Membership.mem F.submodule.toAddSubmonoid f
      ⊢ Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => Set.image ⇑(Polyno …
    -/
    rw [← f.sum_single]
    /-
      case a
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      f : PolynomialModule R M
      hf : Membership.mem F.submodule.toAddSubmonoid f
      ⊢ Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => Set.image ⇑(Polyno …
    -/
    apply AddSubmonoid.sum_mem _ _
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      f : PolynomialModule R M
      hf : Membership.mem F.submodule.toAddSubmonoid f
      ⊢ ∀ (c : Nat), Membership.mem f.support c → Membership.mem (AddSubmonoid.closu …
    -/
    rintro c -
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      f : PolynomialModule R M
      hf : Membership.mem F.submodule.toAddSubmonoid f
      c : Nat
      ⊢ Membership.mem (AddSubmonoid.closure (Set.iUnion fun i => Set.image ⇑(Polyno …
    -/
    exact AddSubmonoid.subset_closure (Set.subset_iUnion _ c <| Set.mem_image_of_mem _ (hf c))
    /-
      🎉 no goals
    -/


theorem submodule_span_single :
    Submodule.span (reesAlgebra I) (⋃ i, single R i '' (F.N i : Set M)) = F.submodule := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    ⊢ Eq (Submodule.span (Subtype fun x => Membership.mem (reesAlgebra I) x) (Set. …
  -/
  rw [← Submodule.span_closure, submodule_closure_single, Submodule.coe_toAddSubmonoid]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    ⊢ Eq (Submodule.span (Subtype fun x => Membership.mem (reesAlgebra I) x) ↑F.su …
  -/
  exact Submodule.span_eq (Filtration.submodule F)
  /-
    🎉 no goals
  -/


theorem submodule_eq_span_le_iff_stable_ge (n₀ : ℕ) :
    F.submodule = Submodule.span _ (⋃ i ≤ n₀, single R i '' (F.N i : Set M)) ↔
      ∀ n ≥ n₀, I • F.N n = F.N (n + 1) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    n₀ : Nat
    ⊢ Iff (Eq F.submodule (Submodule.span (Subtype fun x => Membership.mem (reesAl …
  -/
  rw [← submodule_span_single, ← LE.le.le_iff_eq, Submodule.span_le, Set.iUnion_subset_iff]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    n₀ : Nat
    ⊢ Iff (∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) …
  -/
  swap; · exact Submodule.span_mono (Set.iUnion₂_subset_iUnion _ _)
          /-
            🎉 no goals
          -/
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    I : Ideal R
    F : I.Filtration M
    n₀ : Nat
    ⊢ Iff (∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      ⊢ (∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑(F …
    -/
  · intro H n hn
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      H : ∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑( …
      n : Nat
      hn : GE.ge n n₀
      ⊢ Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
    -/
    refine (F.smul_le n).antisymm ?_
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      H : ∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑( …
      n : Nat
      hn : GE.ge n n₀
      ⊢ LE.le (F.N (HAdd.hAdd n 1)) (HSMul.hSMul I (F.N n))
    -/
    intro x hx
    /-
      case mp
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      H : ∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑( …
      n : Nat
      hn : GE.ge n n₀
      x : M
      hx : Membership.mem (F.N (HAdd.hAdd n 1)) x
      ⊢ Membership.mem (HSMul.hSMul I (F.N n)) x
    -/
    obtain ⟨l, hl⟩ := (Finsupp.mem_span_iff_linearCombination _ _ _).mp (H _ ⟨x, hx, rfl⟩)
    /-
      case mp.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      H : ∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑( …
      n : Nat
      hn : GE.ge n n₀
      x : M
      hx : Membership.mem (F.N (HAdd.hAdd n 1)) x
      l : Finsupp (↑(Set.iUnion fun i => Set.iUnion fun h => Set.image ⇑(PolynomialM …
      hl : Eq ((Finsupp.linearCombination (Subtype fun x => Membership.mem (reesAlge …
      ⊢ Membership.mem (HSMul.hSMul I (F.N n)) x
    -/
    replace hl := congr_arg (fun f : ℕ →₀ M => f (n + 1)) hl
    /-
      case mp.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      H : ∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑( …
      n : Nat
      hn : GE.ge n n₀
      x : M
      hx : Membership.mem (F.N (HAdd.hAdd n 1)) x
      l : Finsupp (↑(Set.iUnion fun i => Set.iUnion fun h => Set.image ⇑(PolynomialM …
      hl : Eq ((fun f => f (HAdd.hAdd n 1)) ((Finsupp.linearCombination (Subtype fun …
      ⊢ Membership.mem (HSMul.hSMul I (F.N n)) x
    -/
    dsimp only at hl
    /-
      case mp.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      H : ∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑( …
      n : Nat
      hn : GE.ge n n₀
      x : M
      hx : Membership.mem (F.N (HAdd.hAdd n 1)) x
      l : Finsupp (↑(Set.iUnion fun i => Set.iUnion fun h => Set.image ⇑(PolynomialM …
      hl : Eq (((Finsupp.linearCombination (Subtype fun x => Membership.mem (reesAlg …
      ⊢ Membership.mem (HSMul.hSMul I (F.N n)) x
    -/
    erw [Finsupp.single_eq_same] at hl
    /-
      case mp.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      H : ∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑( …
      n : Nat
      hn : GE.ge n n₀
      x : M
      hx : Membership.mem (F.N (HAdd.hAdd n 1)) x
      l : Finsupp (↑(Set.iUnion fun i => Set.iUnion fun h => Set.image ⇑(PolynomialM …
      hl : Eq (((Finsupp.linearCombination (Subtype fun x => Membership.mem (reesAlg …
      ⊢ Membership.mem (HSMul.hSMul I (F.N n)) x
    -/
    rw [← hl, Finsupp.linearCombination_apply, Finsupp.sum_apply]
    /-
      case mp.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      H : ∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑( …
      n : Nat
      hn : GE.ge n n₀
      x : M
      hx : Membership.mem (F.N (HAdd.hAdd n 1)) x
      l : Finsupp (↑(Set.iUnion fun i => Set.iUnion fun h => Set.image ⇑(PolynomialM …
      hl : Eq (((Finsupp.linearCombination (Subtype fun x => Membership.mem (reesAlg …
      ⊢ Membership.mem (HSMul.hSMul I (F.N n)) (l.sum fun a₁ b => (HSMul.hSMul b ↑a₁ …
    -/
    apply Submodule.sum_mem _ _
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      H : ∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑( …
      n : Nat
      hn : GE.ge n n₀
      x : M
      hx : Membership.mem (F.N (HAdd.hAdd n 1)) x
      l : Finsupp (↑(Set.iUnion fun i => Set.iUnion fun h => Set.image ⇑(PolynomialM …
      hl : Eq (((Finsupp.linearCombination (Subtype fun x => Membership.mem (reesAlg …
      ⊢ ∀ (c : ↑(Set.iUnion fun i => Set.iUnion fun h => Set.image ⇑(PolynomialModul …
    -/
    rintro ⟨_, _, ⟨n', rfl⟩, _, ⟨hn', rfl⟩, m, hm, rfl⟩ -
    /-
      case mk.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      H : ∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑( …
      n : Nat
      hn : GE.ge n n₀
      x : M
      hx : Membership.mem (F.N (HAdd.hAdd n 1)) x
      l : Finsupp (↑(Set.iUnion fun i => Set.iUnion fun h => Set.image ⇑(PolynomialM …
      hl : Eq (((Finsupp.linearCombination (Subtype fun x => Membership.mem (reesAlg …
      n' : Nat
      hn' : LE.le n' n₀
      m : M
      hm : Membership.mem (↑(F.N n')) m
      ⊢ Membership.mem (HSMul.hSMul I (F.N n)) ((fun a₁ b => (HSMul.hSMul b ↑a₁) (HA …
    -/
    dsimp only [Subtype.coe_mk]
    /-
      case mk.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      H : ∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑( …
      n : Nat
      hn : GE.ge n n₀
      x : M
      hx : Membership.mem (F.N (HAdd.hAdd n 1)) x
      l : Finsupp (↑(Set.iUnion fun i => Set.iUnion fun h => Set.image ⇑(PolynomialM …
      hl : Eq (((Finsupp.linearCombination (Subtype fun x => Membership.mem (reesAlg …
      n' : Nat
      hn' : LE.le n' n₀
      m : M
      hm : Membership.mem (↑(F.N n')) m
      ⊢ Membership.mem (HSMul.hSMul I (F.N n)) ((HSMul.hSMul (l ⟨(PolynomialModule.s …
    -/
    rw [Subalgebra.smul_def, smul_single_apply, if_pos (show n' ≤ n + 1 by omega)]
    /-
      case mk.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      H : ∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑( …
      n : Nat
      hn : GE.ge n n₀
      x : M
      hx : Membership.mem (F.N (HAdd.hAdd n 1)) x
      l : Finsupp (↑(Set.iUnion fun i => Set.iUnion fun h => Set.image ⇑(PolynomialM …
      hl : Eq (((Finsupp.linearCombination (Subtype fun x => Membership.mem (reesAlg …
      n' : Nat
      hn' : LE.le n' n₀
      m : M
      hm : Membership.mem (↑(F.N n')) m
      ⊢ Membership.mem (HSMul.hSMul I (F.N n)) (HSMul.hSMul ((↑(l ⟨(PolynomialModule …
    -/
    have e : n' ≤ n := by omega
    /-
      case mk.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      H : ∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑( …
      n : Nat
      hn : GE.ge n n₀
      x : M
      hx : Membership.mem (F.N (HAdd.hAdd n 1)) x
      l : Finsupp (↑(Set.iUnion fun i => Set.iUnion fun h => Set.image ⇑(PolynomialM …
      hl : Eq (((Finsupp.linearCombination (Subtype fun x => Membership.mem (reesAlg …
      n' : Nat
      hn' : LE.le n' n₀
      m : M
      hm : Membership.mem (↑(F.N n')) m
      e : LE.le n' n
      ⊢ Membership.mem (HSMul.hSMul I (F.N n)) (HSMul.hSMul ((↑(l ⟨(PolynomialModule …
    -/
    have := F.pow_smul_le_pow_smul (n - n') n' 1
    /-
      case mk.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      H : ∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑( …
      n : Nat
      hn : GE.ge n n₀
      x : M
      hx : Membership.mem (F.N (HAdd.hAdd n 1)) x
      l : Finsupp (↑(Set.iUnion fun i => Set.iUnion fun h => Set.image ⇑(PolynomialM …
      hl : Eq (((Finsupp.linearCombination (Subtype fun x => Membership.mem (reesAlg …
      n' : Nat
      hn' : LE.le n' n₀
      m : M
      hm : Membership.mem (↑(F.N n')) m
      e : LE.le n' n
      this : LE.le (HSMul.hSMul (HPow.hPow I (HAdd.hAdd (HSub.hSub n n') 1)) (F.N n' …
      ⊢ Membership.mem (HSMul.hSMul I (F.N n)) (HSMul.hSMul ((↑(l ⟨(PolynomialModule …
    -/
    rw [tsub_add_cancel_of_le e, pow_one, add_comm _ 1, ← add_tsub_assoc_of_le e, add_comm] at this
    /-
      case mk.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      H : ∀ (i : Nat), HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑( …
      n : Nat
      hn : GE.ge n n₀
      x : M
      hx : Membership.mem (F.N (HAdd.hAdd n 1)) x
      l : Finsupp (↑(Set.iUnion fun i => Set.iUnion fun h => Set.image ⇑(PolynomialM …
      hl : Eq (((Finsupp.linearCombination (Subtype fun x => Membership.mem (reesAlg …
      n' : Nat
      hn' : LE.le n' n₀
      m : M
      hm : Membership.mem (↑(F.N n')) m
      e : LE.le n' n
      this : LE.le (HSMul.hSMul (HPow.hPow I (HSub.hSub (HAdd.hAdd n 1) n')) (F.N n' …
      ⊢ Membership.mem (HSMul.hSMul I (F.N n)) (HSMul.hSMul ((↑(l ⟨(PolynomialModule …
    -/
    exact this (Submodule.smul_mem_smul ((l _).2 <| n + 1 - n') hm)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      ⊢ (∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))) …
    -/
  · let F' := Submodule.span (reesAlgebra I) (⋃ i ≤ n₀, single R i '' (F.N i : Set M))
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      F' : Submodule (Subtype fun x => Membership.mem (reesAlgebra I) x) (Polynomial …
      ⊢ (∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))) …
    -/
    intro hF i
    have : ∀ i ≤ n₀, single R i '' (F.N i : Set M) ⊆ F' := by
      -- Porting note: Original proof was
      -- `fun i hi => Set.Subset.trans (Set.subset_iUnion₂ i hi) Submodule.subset_span`
      intro i hi
      refine Set.Subset.trans ?_ Submodule.subset_span
      refine @Set.subset_iUnion₂ _ _ _ (fun i => fun _ => ↑((single R i) '' ((N F i) : Set M))) i ?_
      exact hi
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      F' : Submodule (Subtype fun x => Membership.mem (reesAlgebra I) x) (Polynomial …
      hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
      i : Nat
      this : ∀ (i : Nat), LE.le i n₀ → HasSubset.Subset (Set.image ⇑(PolynomialModul …
      ⊢ HasSubset.Subset (Set.image ⇑(PolynomialModule.single R i) ↑(F.N i)) ↑(Submo …
    -/
    induction' i with j hj
      /-
        case mpr.zero
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        I : Ideal R
        F : I.Filtration M
        n₀ : Nat
        F' : Submodule (Subtype fun x => Membership.mem (reesAlgebra I) x) (Polynomial …
        hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
        this : ∀ (i : Nat), LE.le i n₀ → HasSubset.Subset (Set.image ⇑(PolynomialModul …
        ⊢ HasSubset.Subset (Set.image ⇑(PolynomialModule.single R 0) ↑(F.N 0)) ↑(Submo …
      -/
    · exact this _ (zero_le _)
      /-
        🎉 no goals
      -/
    /-
      case mpr.succ
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      F' : Submodule (Subtype fun x => Membership.mem (reesAlgebra I) x) (Polynomial …
      hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
      this : ∀ (i : Nat), LE.le i n₀ → HasSubset.Subset (Set.image ⇑(PolynomialModul …
      j : Nat
      hj : HasSubset.Subset (Set.image ⇑(PolynomialModule.single R j) ↑(F.N j)) ↑(Su …
      ⊢ HasSubset.Subset (Set.image ⇑(PolynomialModule.single R (HAdd.hAdd j 1)) ↑(F …
    -/
    by_cases hj' : j.succ ≤ n₀
      /-
        case pos
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        I : Ideal R
        F : I.Filtration M
        n₀ : Nat
        F' : Submodule (Subtype fun x => Membership.mem (reesAlgebra I) x) (Polynomial …
        hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
        this : ∀ (i : Nat), LE.le i n₀ → HasSubset.Subset (Set.image ⇑(PolynomialModul …
        j : Nat
        hj : HasSubset.Subset (Set.image ⇑(PolynomialModule.single R j) ↑(F.N j)) ↑(Su …
        hj' : LE.le j.succ n₀
        ⊢ HasSubset.Subset (Set.image ⇑(PolynomialModule.single R (HAdd.hAdd j 1)) ↑(F …
      -/
    · exact this _ hj'
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      F' : Submodule (Subtype fun x => Membership.mem (reesAlgebra I) x) (Polynomial …
      hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
      this : ∀ (i : Nat), LE.le i n₀ → HasSubset.Subset (Set.image ⇑(PolynomialModul …
      j : Nat
      hj : HasSubset.Subset (Set.image ⇑(PolynomialModule.single R j) ↑(F.N j)) ↑(Su …
      hj' : Not (LE.le j.succ n₀)
      ⊢ HasSubset.Subset (Set.image ⇑(PolynomialModule.single R (HAdd.hAdd j 1)) ↑(F …
    -/
    simp only [not_le, Nat.lt_succ_iff] at hj'
    /-
      case neg
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      F' : Submodule (Subtype fun x => Membership.mem (reesAlgebra I) x) (Polynomial …
      hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
      this : ∀ (i : Nat), LE.le i n₀ → HasSubset.Subset (Set.image ⇑(PolynomialModul …
      j : Nat
      hj : HasSubset.Subset (Set.image ⇑(PolynomialModule.single R j) ↑(F.N j)) ↑(Su …
      hj' : LE.le n₀ j
      ⊢ HasSubset.Subset (Set.image ⇑(PolynomialModule.single R (HAdd.hAdd j 1)) ↑(F …
    -/
    rw [← hF _ hj']
    /-
      case neg
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      F' : Submodule (Subtype fun x => Membership.mem (reesAlgebra I) x) (Polynomial …
      hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
      this : ∀ (i : Nat), LE.le i n₀ → HasSubset.Subset (Set.image ⇑(PolynomialModul …
      j : Nat
      hj : HasSubset.Subset (Set.image ⇑(PolynomialModule.single R j) ↑(F.N j)) ↑(Su …
      hj' : LE.le n₀ j
      ⊢ HasSubset.Subset (Set.image ⇑(PolynomialModule.single R (HAdd.hAdd j 1)) ↑(H …
    -/
    rintro _ ⟨m, hm, rfl⟩
    /-
      case neg.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝² : CommRing R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      I : Ideal R
      F : I.Filtration M
      n₀ : Nat
      F' : Submodule (Subtype fun x => Membership.mem (reesAlgebra I) x) (Polynomial …
      hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
      this : ∀ (i : Nat), LE.le i n₀ → HasSubset.Subset (Set.image ⇑(PolynomialModul …
      j : Nat
      hj : HasSubset.Subset (Set.image ⇑(PolynomialModule.single R j) ↑(F.N j)) ↑(Su …
      hj' : LE.le n₀ j
      m : M
      hm : Membership.mem (↑(HSMul.hSMul I (F.N j))) m
      ⊢ Membership.mem (↑(Submodule.span (Subtype fun x => Membership.mem (reesAlgeb …
    -/
    refine Submodule.smul_induction_on hm (fun r hr m' hm' => ?_) (fun x y hx hy => ?_)
      /-
        case neg.intro.intro.refine_1
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        I : Ideal R
        F : I.Filtration M
        n₀ : Nat
        F' : Submodule (Subtype fun x => Membership.mem (reesAlgebra I) x) (Polynomial …
        hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
        this : ∀ (i : Nat), LE.le i n₀ → HasSubset.Subset (Set.image ⇑(PolynomialModul …
        j : Nat
        hj : HasSubset.Subset (Set.image ⇑(PolynomialModule.single R j) ↑(F.N j)) ↑(Su …
        hj' : LE.le n₀ j
        m : M
        hm : Membership.mem (↑(HSMul.hSMul I (F.N j))) m
        r : R
        hr : Membership.mem I r
        m' : M
        hm' : Membership.mem (F.N j) m'
        ⊢ Membership.mem (↑(Submodule.span (Subtype fun x => Membership.mem (reesAlgeb …
      -/
    · rw [add_comm, ← monomial_smul_single]
      exact F'.smul_mem
        ⟨_, reesAlgebra.monomial_mem.mpr (by rwa [pow_one])⟩ (hj <| Set.mem_image_of_mem _ hm')
      /-
        case neg.intro.intro.refine_2
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        I : Ideal R
        F : I.Filtration M
        n₀ : Nat
        F' : Submodule (Subtype fun x => Membership.mem (reesAlgebra I) x) (Polynomial …
        hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
        this : ∀ (i : Nat), LE.le i n₀ → HasSubset.Subset (Set.image ⇑(PolynomialModul …
        j : Nat
        hj : HasSubset.Subset (Set.image ⇑(PolynomialModule.single R j) ↑(F.N j)) ↑(Su …
        hj' : LE.le n₀ j
        m : M
        hm : Membership.mem (↑(HSMul.hSMul I (F.N j))) m
        x y : M
        hx : Membership.mem (↑(Submodule.span (Subtype fun x => Membership.mem (reesAl …
        hy : Membership.mem (↑(Submodule.span (Subtype fun x => Membership.mem (reesAl …
        ⊢ Membership.mem (↑(Submodule.span (Subtype fun x => Membership.mem (reesAlgeb …
      -/
    · rw [map_add]
      /-
        case neg.intro.intro.refine_2
        R : Type u_1
        M : Type u_2
        inst✝² : CommRing R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        I : Ideal R
        F : I.Filtration M
        n₀ : Nat
        F' : Submodule (Subtype fun x => Membership.mem (reesAlgebra I) x) (Polynomial …
        hF : ∀ (n : Nat), GE.ge n n₀ → Eq (HSMul.hSMul I (F.N n)) (F.N (HAdd.hAdd n 1))
        this : ∀ (i : Nat), LE.le i n₀ → HasSubset.Subset (Set.image ⇑(PolynomialModul …
        j : Nat
        hj : HasSubset.Subset (Set.image ⇑(PolynomialModule.single R j) ↑(F.N j)) ↑(Su …
        hj' : LE.le n₀ j
        m : M
        hm : Membership.mem (↑(HSMul.hSMul I (F.N j))) m
        x y : M
        hx : Membership.mem (↑(Submodule.span (Subtype fun x => Membership.mem (reesAl …
        hy : Membership.mem (↑(Submodule.span (Subtype fun x => Membership.mem (reesAl …
        ⊢ Membership.mem (↑(Submodule.span (Subtype fun x => Membership.mem (reesAlgeb …
      -/
      exact F'.add_mem hx hy
      /-
        🎉 no goals
      -/


/-- If the components of a filtration are finitely generated, then the filtration is stable iff
its associated submodule of is finitely generated. -/
theorem submodule_fg_iff_stable (hF' : ∀ i, (F.N i).FG) : F.submodule.FG ↔ F.Stable := by
  classical
  delta Ideal.Filtration.Stable
  simp_rw [← F.submodule_eq_span_le_iff_stable_ge]
  constructor
  · rintro H
    refine H.stabilizes_of_iSup_eq
        ⟨fun n₀ => Submodule.span _ (⋃ (i : ℕ) (_ : i ≤ n₀), single R i '' ↑(F.N i)), ?_⟩ ?_
    · intro n m e
      rw [Submodule.span_le, Set.iUnion₂_subset_iff]
      intro i hi
      refine Set.Subset.trans ?_ Submodule.subset_span
      refine @Set.subset_iUnion₂ _ _ _ (fun i => fun _ => ↑((single R i) '' ((N F i) : Set M))) i ?_
      exact hi.trans e
    · dsimp
      rw [← Submodule.span_iUnion, ← submodule_span_single]
      congr 1
      ext
      simp only [Set.mem_iUnion, Set.mem_image, SetLike.mem_coe, exists_prop]
      constructor
      · rintro ⟨-, i, -, e⟩; exact ⟨i, e⟩
      · rintro ⟨i, e⟩; exact ⟨i, i, le_refl i, e⟩
  · rintro ⟨n, hn⟩
    rw [hn]
    simp_rw [Submodule.span_iUnion₂, ← Finset.mem_range_succ_iff, iSup_subtype']
    apply Submodule.fg_iSup
    rintro ⟨i, hi⟩
    obtain ⟨s, hs⟩ := hF' i
    have : Submodule.span (reesAlgebra I) (s.image (lsingle R i) : Set (PolynomialModule R M)) =
        Submodule.span _ (single R i '' (F.N i : Set M)) := by
      rw [Finset.coe_image, ← Submodule.span_span_of_tower R, ← Submodule.map_span, hs]; rfl
    rw [Subtype.coe_mk, ← this]
    exact ⟨_, rfl⟩


theorem Stable.of_le [IsNoetherianRing R] [Module.Finite R M] (hF : F.Stable)
    {F' : I.Filtration M} (hf : F' ≤ F) : F'.Stable := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    I : Ideal R
    F : I.Filtration M
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R M
    hF : F.Stable
    F' : I.Filtration M
    hf : LE.le F' F
    ⊢ F'.Stable
  -/
  rw [← submodule_fg_iff_stable] at hF ⊢
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    I : Ideal R
    F : I.Filtration M
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R M
    hF : F.submodule.FG
    F' : I.Filtration M
    hf : LE.le F' F
    ⊢ F'.submodule.FG
  -/
  any_goals intro i; exact IsNoetherian.noetherian _
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    I : Ideal R
    F : I.Filtration M
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R M
    hF : F.submodule.FG
    F' : I.Filtration M
    hf : LE.le F' F
    ⊢ F'.submodule.FG
  -/
  have := isNoetherian_of_fg_of_noetherian _ hF
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    I : Ideal R
    F : I.Filtration M
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R M
    hF : F.submodule.FG
    F' : I.Filtration M
    hf : LE.le F' F
    this : IsNoetherian (Subtype fun x => Membership.mem (reesAlgebra I) x) (Subty …
    ⊢ F'.submodule.FG
  -/
  rw [isNoetherian_submodule] at this
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    I : Ideal R
    F : I.Filtration M
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R M
    hF : F.submodule.FG
    F' : I.Filtration M
    hf : LE.le F' F
    this : ∀ (s : Submodule (Subtype fun x => Membership.mem (reesAlgebra I) x) (P …
    ⊢ F'.submodule.FG
  -/
  exact this _ (OrderHomClass.mono (submoduleInfHom M I) hf)
  /-
    🎉 no goals
  -/


theorem Stable.inter_right [IsNoetherianRing R] [Module.Finite R M] (hF : F.Stable) :
    (F ⊓ F').Stable :=
  hF.of_le inf_le_left


theorem Stable.inter_left [IsNoetherianRing R] [Module.Finite R M] (hF : F.Stable) :
    (F' ⊓ F).Stable :=
  hF.of_le inf_le_right


/-- **Artin-Rees lemma** -/
theorem Ideal.exists_pow_inf_eq_pow_smul [IsNoetherianRing R] [Module.Finite R M]
    (N : Submodule R M) : ∃ k : ℕ, ∀ n ≥ k, I ^ n • ⊤ ⊓ N = I ^ (n - k) • (I ^ k • ⊤ ⊓ N) :=
  ((I.stableFiltration_stable ⊤).inter_right (I.trivialFiltration N)).exists_pow_smul_eq_of_ge


theorem Ideal.mem_iInf_smul_pow_eq_bot_iff [IsNoetherianRing R] [Module.Finite R M] (x : M) :
    x ∈ (⨅ i : ℕ, I ^ i • ⊤ : Submodule R M) ↔ ∃ r : I, (r : R) • x = x := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    I : Ideal R
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R M
    x : M
    ⊢ Iff (Membership.mem (iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top) x) ( …
  -/
  let N := (⨅ i : ℕ, I ^ i • ⊤ : Submodule R M)
  have hN : ∀ k, (I.stableFiltration ⊤ ⊓ I.trivialFiltration N).N k = N :=
    fun k => inf_eq_right.mpr ((iInf_le _ k).trans <| le_of_eq <| by simp)
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    I : Ideal R
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R M
    x : M
    N : Submodule R M := iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top
    hN : ∀ (k : Nat), Eq ((Min.min (I.stableFiltration Top.top) (I.trivialFiltrati …
    ⊢ Iff (Membership.mem (iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top) x) ( …
  -/
  constructor
  · obtain ⟨r, hr₁, hr₂⟩ :=
      Submodule.exists_mem_and_smul_eq_self_of_fg_of_le_smul I N (IsNoetherian.noetherian N) (by
        obtain ⟨k, hk⟩ := (I.stableFiltration_stable ⊤).inter_right (I.trivialFiltration N)
        have := hk k (le_refl _)
        rw [hN, hN] at this
        exact le_of_eq this.symm)
    /-
      case mp.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      I : Ideal R
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R M
      x : M
      N : Submodule R M := iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top
      hN : ∀ (k : Nat), Eq ((Min.min (I.stableFiltration Top.top) (I.trivialFiltrati …
      r : R
      hr₁ : Membership.mem I r
      hr₂ : ∀ (n : M), Membership.mem N n → Eq (HSMul.hSMul r n) n
      ⊢ Membership.mem (iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top) x → Exist …
    -/
    intro H
    /-
      case mp.intro.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      I : Ideal R
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R M
      x : M
      N : Submodule R M := iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top
      hN : ∀ (k : Nat), Eq ((Min.min (I.stableFiltration Top.top) (I.trivialFiltrati …
      r : R
      hr₁ : Membership.mem I r
      hr₂ : ∀ (n : M), Membership.mem N n → Eq (HSMul.hSMul r n) n
      H : Membership.mem (iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top) x
      ⊢ Exists fun r => Eq (HSMul.hSMul (↑r) x) x
    -/
    exact ⟨⟨r, hr₁⟩, hr₂ _ H⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      I : Ideal R
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R M
      x : M
      N : Submodule R M := iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top
      hN : ∀ (k : Nat), Eq ((Min.min (I.stableFiltration Top.top) (I.trivialFiltrati …
      ⊢ (Exists fun r => Eq (HSMul.hSMul (↑r) x) x) → Membership.mem (iInf fun i =>  …
    -/
  · rintro ⟨r, eq⟩
    /-
      case mpr.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      I : Ideal R
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R M
      x : M
      N : Submodule R M := iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top
      hN : ∀ (k : Nat), Eq ((Min.min (I.stableFiltration Top.top) (I.trivialFiltrati …
      r : Subtype fun x => Membership.mem I x
      eq : Eq (HSMul.hSMul (↑r) x) x
      ⊢ Membership.mem (iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top) x
    -/
    rw [Submodule.mem_iInf]
    /-
      case mpr.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      I : Ideal R
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R M
      x : M
      N : Submodule R M := iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top
      hN : ∀ (k : Nat), Eq ((Min.min (I.stableFiltration Top.top) (I.trivialFiltrati …
      r : Subtype fun x => Membership.mem I x
      eq : Eq (HSMul.hSMul (↑r) x) x
      ⊢ ∀ (i : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I i) Top.top) x
    -/
    intro i
    /-
      case mpr.intro
      R : Type u_1
      M : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      I : Ideal R
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R M
      x : M
      N : Submodule R M := iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top
      hN : ∀ (k : Nat), Eq ((Min.min (I.stableFiltration Top.top) (I.trivialFiltrati …
      r : Subtype fun x => Membership.mem I x
      eq : Eq (HSMul.hSMul (↑r) x) x
      i : Nat
      ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I i) Top.top) x
    -/
    induction' i with i hi
      /-
        case mpr.intro.zero
        R : Type u_1
        M : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        I : Ideal R
        inst✝¹ : IsNoetherianRing R
        inst✝ : Module.Finite R M
        x : M
        N : Submodule R M := iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top
        hN : ∀ (k : Nat), Eq ((Min.min (I.stableFiltration Top.top) (I.trivialFiltrati …
        r : Subtype fun x => Membership.mem I x
        eq : Eq (HSMul.hSMul (↑r) x) x
        ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I 0) Top.top) x
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.succ
        R : Type u_1
        M : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        I : Ideal R
        inst✝¹ : IsNoetherianRing R
        inst✝ : Module.Finite R M
        x : M
        N : Submodule R M := iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top
        hN : ∀ (k : Nat), Eq ((Min.min (I.stableFiltration Top.top) (I.trivialFiltrati …
        r : Subtype fun x => Membership.mem I x
        eq : Eq (HSMul.hSMul (↑r) x) x
        i : Nat
        hi : Membership.mem (HSMul.hSMul (HPow.hPow I i) Top.top) x
        ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd i 1)) Top.top) x
      -/
    · rw [add_comm, pow_add, ← smul_smul, pow_one, ← eq]
      /-
        case mpr.intro.succ
        R : Type u_1
        M : Type u_2
        inst✝⁴ : CommRing R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        I : Ideal R
        inst✝¹ : IsNoetherianRing R
        inst✝ : Module.Finite R M
        x : M
        N : Submodule R M := iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top
        hN : ∀ (k : Nat), Eq ((Min.min (I.stableFiltration Top.top) (I.trivialFiltrati …
        r : Subtype fun x => Membership.mem I x
        eq : Eq (HSMul.hSMul (↑r) x) x
        i : Nat
        hi : Membership.mem (HSMul.hSMul (HPow.hPow I i) Top.top) x
        ⊢ Membership.mem (HSMul.hSMul I (HSMul.hSMul (HPow.hPow I i) Top.top)) (HSMul. …
      -/
      exact Submodule.smul_mem_smul r.prop hi
      /-
        🎉 no goals
      -/


theorem Ideal.iInf_pow_smul_eq_bot_of_isLocalRing [IsNoetherianRing R] [IsLocalRing R]
    [Module.Finite R M] (h : I ≠ ⊤) : (⨅ i : ℕ, I ^ i • ⊤ : Submodule R M) = ⊥ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    I : Ideal R
    inst✝² : IsNoetherianRing R
    inst✝¹ : IsLocalRing R
    inst✝ : Module.Finite R M
    h : Ne I Top.top
    ⊢ Eq (iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top) Bot.bot
  -/
  rw [eq_bot_iff]
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    I : Ideal R
    inst✝² : IsNoetherianRing R
    inst✝¹ : IsLocalRing R
    inst✝ : Module.Finite R M
    h : Ne I Top.top
    ⊢ LE.le (iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top) Bot.bot
  -/
  intro x hx
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    I : Ideal R
    inst✝² : IsNoetherianRing R
    inst✝¹ : IsLocalRing R
    inst✝ : Module.Finite R M
    h : Ne I Top.top
    x : M
    hx : Membership.mem (iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top) x
    ⊢ Membership.mem Bot.bot x
  -/
  obtain ⟨r, hr⟩ := (I.mem_iInf_smul_pow_eq_bot_iff x).mp hx
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    I : Ideal R
    inst✝² : IsNoetherianRing R
    inst✝¹ : IsLocalRing R
    inst✝ : Module.Finite R M
    h : Ne I Top.top
    x : M
    hx : Membership.mem (iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top) x
    r : Subtype fun x => Membership.mem I x
    hr : Eq (HSMul.hSMul (↑r) x) x
    ⊢ Membership.mem Bot.bot x
  -/
  have := IsLocalRing.isUnit_one_sub_self_of_mem_nonunits _ (IsLocalRing.le_maximalIdeal h r.prop)
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    I : Ideal R
    inst✝² : IsNoetherianRing R
    inst✝¹ : IsLocalRing R
    inst✝ : Module.Finite R M
    h : Ne I Top.top
    x : M
    hx : Membership.mem (iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top) x
    r : Subtype fun x => Membership.mem I x
    hr : Eq (HSMul.hSMul (↑r) x) x
    this : IsUnit (HSub.hSub 1 ↑r)
    ⊢ Membership.mem Bot.bot x
  -/
  apply this.smul_left_cancel.mp
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    I : Ideal R
    inst✝² : IsNoetherianRing R
    inst✝¹ : IsLocalRing R
    inst✝ : Module.Finite R M
    h : Ne I Top.top
    x : M
    hx : Membership.mem (iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top) x
    r : Subtype fun x => Membership.mem I x
    hr : Eq (HSMul.hSMul (↑r) x) x
    this : IsUnit (HSub.hSub 1 ↑r)
    ⊢ Eq (HSMul.hSMul (HSub.hSub 1 ↑r) x) (HSMul.hSMul (HSub.hSub 1 ↑r) 0)
  -/
  simp [sub_smul, hr]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-12")]
alias Ideal.iInf_pow_smul_eq_bot_of_localRing := Ideal.iInf_pow_smul_eq_bot_of_isLocalRing


/-- **Krull's intersection theorem** for noetherian local rings. -/
theorem Ideal.iInf_pow_eq_bot_of_isLocalRing [IsNoetherianRing R] [IsLocalRing R] (h : I ≠ ⊤) :
    ⨅ i : ℕ, I ^ i = ⊥ := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    inst✝¹ : IsNoetherianRing R
    inst✝ : IsLocalRing R
    h : Ne I Top.top
    ⊢ Eq (iInf fun i => HPow.hPow I i) Bot.bot
  -/
  convert I.iInf_pow_smul_eq_bot_of_isLocalRing (M := R) h
  /-
    case h.e'_2.h.e'_4.h
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    inst✝¹ : IsNoetherianRing R
    inst✝ : IsLocalRing R
    h : Ne I Top.top
    x✝ : Nat
    ⊢ Eq (HPow.hPow I x✝) (HSMul.hSMul (HPow.hPow I x✝) Top.top)
  -/
  ext i
  /-
    case h.e'_2.h.e'_4.h.h
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    inst✝¹ : IsNoetherianRing R
    inst✝ : IsLocalRing R
    h : Ne I Top.top
    x✝ : Nat
    i : R
    ⊢ Iff (Membership.mem (HPow.hPow I x✝) i) (Membership.mem (HSMul.hSMul (HPow.h …
  -/
  rw [smul_eq_mul, ← Ideal.one_eq_top, mul_one]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-12")]
alias Ideal.iInf_pow_eq_bot_of_localRing := Ideal.iInf_pow_eq_bot_of_isLocalRing


/-- Also see `Ideal.isIdempotentElem_iff_eq_bot_or_top` for integral domains. -/
theorem Ideal.isIdempotentElem_iff_eq_bot_or_top_of_isLocalRing {R} [CommRing R]
    [IsNoetherianRing R] [IsLocalRing R] (I : Ideal R) :
    IsIdempotentElem I ↔ I = ⊥ ∨ I = ⊤ := by
  /-
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : IsNoetherianRing R
    inst✝ : IsLocalRing R
    I : Ideal R
    ⊢ Iff (IsIdempotentElem I) (Or (Eq I Bot.bot) (Eq I Top.top))
  -/
  constructor
    /-
      case mp
      R : Type u_3
      inst✝² : CommRing R
      inst✝¹ : IsNoetherianRing R
      inst✝ : IsLocalRing R
      I : Ideal R
      ⊢ IsIdempotentElem I → Or (Eq I Bot.bot) (Eq I Top.top)
    -/
  · intro H
    /-
      case mp
      R : Type u_3
      inst✝² : CommRing R
      inst✝¹ : IsNoetherianRing R
      inst✝ : IsLocalRing R
      I : Ideal R
      H : IsIdempotentElem I
      ⊢ Or (Eq I Bot.bot) (Eq I Top.top)
    -/
    by_cases I = ⊤; · exact Or.inr ‹_›
                      /-
                        🎉 no goals
                      -/
    /-
      case neg
      R : Type u_3
      inst✝² : CommRing R
      inst✝¹ : IsNoetherianRing R
      inst✝ : IsLocalRing R
      I : Ideal R
      H : IsIdempotentElem I
      h✝ : Not (Eq I Top.top)
      ⊢ Or (Eq I Bot.bot) (Eq I Top.top)
    -/
    refine Or.inl (eq_bot_iff.mpr ?_)
    /-
      case neg
      R : Type u_3
      inst✝² : CommRing R
      inst✝¹ : IsNoetherianRing R
      inst✝ : IsLocalRing R
      I : Ideal R
      H : IsIdempotentElem I
      h✝ : Not (Eq I Top.top)
      ⊢ LE.le I Bot.bot
    -/
    rw [← Ideal.iInf_pow_eq_bot_of_isLocalRing I ‹_›]
    /-
      case neg
      R : Type u_3
      inst✝² : CommRing R
      inst✝¹ : IsNoetherianRing R
      inst✝ : IsLocalRing R
      I : Ideal R
      H : IsIdempotentElem I
      h✝ : Not (Eq I Top.top)
      ⊢ LE.le I (iInf fun i => HPow.hPow I i)
    -/
    apply le_iInf
    /-
      case neg.h
      R : Type u_3
      inst✝² : CommRing R
      inst✝¹ : IsNoetherianRing R
      inst✝ : IsLocalRing R
      I : Ideal R
      H : IsIdempotentElem I
      h✝ : Not (Eq I Top.top)
      ⊢ ∀ (i : Nat), LE.le I (HPow.hPow I i)
    -/
                     /-
                       🎉 no goals
                     -/
    rintro (_|n) <;> simp [H.pow_succ_eq]
                     /-
                       🎉 no goals
                     -/
    /-
      case mpr
      R : Type u_3
      inst✝² : CommRing R
      inst✝¹ : IsNoetherianRing R
      inst✝ : IsLocalRing R
      I : Ideal R
      ⊢ Or (Eq I Bot.bot) (Eq I Top.top) → IsIdempotentElem I
    -/
                           /-
                             🎉 no goals
                           -/
  · rintro (rfl | rfl) <;> simp [IsIdempotentElem]
                           /-
                             🎉 no goals
                           -/


@[deprecated (since := "2024-11-12")]
alias Ideal.isIdempotentElem_iff_eq_bot_or_top_of_localRing :=
  Ideal.isIdempotentElem_iff_eq_bot_or_top_of_isLocalRing


/-- **Krull's intersection theorem** for noetherian domains. -/
theorem Ideal.iInf_pow_eq_bot_of_isDomain [IsNoetherianRing R] [IsDomain R] (h : I ≠ ⊤) :
    ⨅ i : ℕ, I ^ i = ⊥ := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    inst✝¹ : IsNoetherianRing R
    inst✝ : IsDomain R
    h : Ne I Top.top
    ⊢ Eq (iInf fun i => HPow.hPow I i) Bot.bot
  -/
  rw [eq_bot_iff]
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    inst✝¹ : IsNoetherianRing R
    inst✝ : IsDomain R
    h : Ne I Top.top
    ⊢ LE.le (iInf fun i => HPow.hPow I i) Bot.bot
  -/
  intro x hx
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    inst✝¹ : IsNoetherianRing R
    inst✝ : IsDomain R
    h : Ne I Top.top
    x : R
    hx : Membership.mem (iInf fun i => HPow.hPow I i) x
    ⊢ Membership.mem Bot.bot x
  -/
  by_contra hx'
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    inst✝¹ : IsNoetherianRing R
    inst✝ : IsDomain R
    h : Ne I Top.top
    x : R
    hx : Membership.mem (iInf fun i => HPow.hPow I i) x
    hx' : Not (Membership.mem Bot.bot x)
    ⊢ False
  -/
  have := Ideal.mem_iInf_smul_pow_eq_bot_iff I x
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    inst✝¹ : IsNoetherianRing R
    inst✝ : IsDomain R
    h : Ne I Top.top
    x : R
    hx : Membership.mem (iInf fun i => HPow.hPow I i) x
    hx' : Not (Membership.mem Bot.bot x)
    this : Iff (Membership.mem (iInf fun i => HSMul.hSMul (HPow.hPow I i) Top.top) …
    ⊢ False
  -/
  simp_rw [smul_eq_mul, ← Ideal.one_eq_top, mul_one] at this
  /-
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    inst✝¹ : IsNoetherianRing R
    inst✝ : IsDomain R
    h : Ne I Top.top
    x : R
    hx : Membership.mem (iInf fun i => HPow.hPow I i) x
    hx' : Not (Membership.mem Bot.bot x)
    this : Iff (Membership.mem (iInf fun i => HPow.hPow I i) x) (Exists fun r => E …
    ⊢ False
  -/
  obtain ⟨r, hr⟩ := this.mp hx
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    inst✝¹ : IsNoetherianRing R
    inst✝ : IsDomain R
    h : Ne I Top.top
    x : R
    hx : Membership.mem (iInf fun i => HPow.hPow I i) x
    hx' : Not (Membership.mem Bot.bot x)
    this : Iff (Membership.mem (iInf fun i => HPow.hPow I i) x) (Exists fun r => E …
    r : Subtype fun x => Membership.mem I x
    hr : Eq (HMul.hMul (↑r) x) x
    ⊢ False
  -/
  have := mul_right_cancel₀ hx' (hr.trans (one_mul x).symm)
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    I : Ideal R
    inst✝¹ : IsNoetherianRing R
    inst✝ : IsDomain R
    h : Ne I Top.top
    x : R
    hx : Membership.mem (iInf fun i => HPow.hPow I i) x
    hx' : Not (Membership.mem Bot.bot x)
    this✝ : Iff (Membership.mem (iInf fun i => HPow.hPow I i) x) (Exists fun r =>  …
    r : Subtype fun x => Membership.mem I x
    hr : Eq (HMul.hMul (↑r) x) x
    this : Eq (↑r) 1
    ⊢ False
  -/
  exact I.eq_top_iff_one.not.mp h (this ▸ r.prop)
  /-
    🎉 no goals
  -/

