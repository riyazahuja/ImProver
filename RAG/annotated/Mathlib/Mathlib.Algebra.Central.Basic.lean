@[simp]
lemma center_eq_bot : Subalgebra.center K D = ⊥ := eq_bot_iff.2 IsCentral.out


variable {D} in
lemma mem_center_iff {x : D} : x ∈ Subalgebra.center K D ↔ ∃ (a : K), x = algebraMap K D a := by
  /-
    K : Type u
    inst✝³ : CommSemiring K
    D : Type v
    inst✝² : Semiring D
    inst✝¹ : Algebra K D
    inst✝ : Algebra.IsCentral K D
    x : D
    ⊢ Iff (Membership.mem (Subalgebra.center K D) x) (Exists fun a => Eq x ((algeb …
  -/
  rw [center_eq_bot, Algebra.mem_bot]
  /-
    K : Type u
    inst✝³ : CommSemiring K
    D : Type v
    inst✝² : Semiring D
    inst✝¹ : Algebra K D
    inst✝ : Algebra.IsCentral K D
    x : D
    ⊢ Iff (Membership.mem (Set.range ⇑(algebraMap K D)) x) (Exists fun a => Eq x ( …
  -/
  simp [eq_comm]
  /-
    🎉 no goals
  -/


instance self : IsCentral K K where
              /-
                K : Type u
                inst✝³ : CommSemiring K
                D : Type v
                inst✝² : Semiring D
                inst✝¹ : Algebra K D
                inst✝ : Algebra.IsCentral K D
                x : K
                ⊢ Membership.mem (Subalgebra.center K K) x → Membership.mem Bot.bot x
              -/
  out x := by simp [Algebra.mem_bot]
              /-
                🎉 no goals
              -/


lemma baseField_essentially_unique
    (k K D : Type*) [Field k] [Field K] [Ring D] [Nontrivial D]
    [Algebra k K] [Algebra K D] [Algebra k D] [IsScalarTower k K D]
    [IsCentral k D] :
    Function.Bijective (algebraMap k K) := by
  haveI : IsCentral K D :=
  { out := fun x ↦ show x ∈ Subalgebra.center k D → _ by
      simp only [center_eq_bot, mem_bot, Set.mem_range, forall_exists_index]
      rintro x rfl
      exact  ⟨algebraMap k K x, by simp [algebraMap_eq_smul_one, smul_assoc]⟩ }
  /-
    k : Type u_1
    K : Type u_2
    D : Type u_3
    inst✝⁸ : Field k
    inst✝⁷ : Field K
    inst✝⁶ : Ring D
    inst✝⁵ : Nontrivial D
    inst✝⁴ : Algebra k K
    inst✝³ : Algebra K D
    inst✝² : Algebra k D
    inst✝¹ : IsScalarTower k K D
    inst✝ : Algebra.IsCentral k D
    this : Algebra.IsCentral K D
    ⊢ Function.Bijective ⇑(algebraMap k K)
  -/
  refine ⟨NoZeroSMulDivisors.algebraMap_injective k K, fun x => ?_⟩
  /-
    k : Type u_1
    K : Type u_2
    D : Type u_3
    inst✝⁸ : Field k
    inst✝⁷ : Field K
    inst✝⁶ : Ring D
    inst✝⁵ : Nontrivial D
    inst✝⁴ : Algebra k K
    inst✝³ : Algebra K D
    inst✝² : Algebra k D
    inst✝¹ : IsScalarTower k K D
    inst✝ : Algebra.IsCentral k D
    this : Algebra.IsCentral K D
    x : K
    ⊢ Exists fun a => Eq ((algebraMap k K) a) x
  -/
  have H : algebraMap K D x ∈ (Subalgebra.center K D : Set D) := Subalgebra.algebraMap_mem _ _
  /-
    k : Type u_1
    K : Type u_2
    D : Type u_3
    inst✝⁸ : Field k
    inst✝⁷ : Field K
    inst✝⁶ : Ring D
    inst✝⁵ : Nontrivial D
    inst✝⁴ : Algebra k K
    inst✝³ : Algebra K D
    inst✝² : Algebra k D
    inst✝¹ : IsScalarTower k K D
    inst✝ : Algebra.IsCentral k D
    this : Algebra.IsCentral K D
    x : K
    H : Membership.mem (↑(Subalgebra.center K D)) ((algebraMap K D) x)
    ⊢ Exists fun a => Eq ((algebraMap k K) a) x
  -/
  rw [show (Subalgebra.center K D : Set D) = Subalgebra.center k D by rfl] at H
  /-
    k : Type u_1
    K : Type u_2
    D : Type u_3
    inst✝⁸ : Field k
    inst✝⁷ : Field K
    inst✝⁶ : Ring D
    inst✝⁵ : Nontrivial D
    inst✝⁴ : Algebra k K
    inst✝³ : Algebra K D
    inst✝² : Algebra k D
    inst✝¹ : IsScalarTower k K D
    inst✝ : Algebra.IsCentral k D
    this : Algebra.IsCentral K D
    x : K
    H : Membership.mem (↑(Subalgebra.center k D)) ((algebraMap K D) x)
    ⊢ Exists fun a => Eq ((algebraMap k K) a) x
  -/
  simp only [center_eq_bot, coe_bot, Set.mem_range] at H
  /-
    k : Type u_1
    K : Type u_2
    D : Type u_3
    inst✝⁸ : Field k
    inst✝⁷ : Field K
    inst✝⁶ : Ring D
    inst✝⁵ : Nontrivial D
    inst✝⁴ : Algebra k K
    inst✝³ : Algebra K D
    inst✝² : Algebra k D
    inst✝¹ : IsScalarTower k K D
    inst✝ : Algebra.IsCentral k D
    this : Algebra.IsCentral K D
    x : K
    H : Exists fun y => Eq ((algebraMap k D) y) ((algebraMap K D) x)
    ⊢ Exists fun a => Eq ((algebraMap k K) a) x
  -/
  obtain ⟨x', H⟩ := H
  /-
    case intro
    k : Type u_1
    K : Type u_2
    D : Type u_3
    inst✝⁸ : Field k
    inst✝⁷ : Field K
    inst✝⁶ : Ring D
    inst✝⁵ : Nontrivial D
    inst✝⁴ : Algebra k K
    inst✝³ : Algebra K D
    inst✝² : Algebra k D
    inst✝¹ : IsScalarTower k K D
    inst✝ : Algebra.IsCentral k D
    this : Algebra.IsCentral K D
    x : K
    x' : k
    H : Eq ((algebraMap k D) x') ((algebraMap K D) x)
    ⊢ Exists fun a => Eq ((algebraMap k K) a) x
  -/
  exact ⟨x', (algebraMap K D).injective <| by simp [← H, algebraMap_eq_smul_one]⟩
  /-
    🎉 no goals
  -/


