/-- **Ruzsa's covering lemma**. -/
@[to_additive "**Ruzsa's covering lemma**"]
theorem ruzsa_covering_mul (hB : B.Nonempty) (hK : #(A * B) ≤ K * #B) :
    ∃ F ⊆ A, #F ≤ K ∧ A ⊆ F * (B / B) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    K : Real
    inst✝ : DecidableEq G
    A B : Finset G
    hB : B.Nonempty
    hK : LE.le (↑(HMul.hMul A B).card) (HMul.hMul K ↑B.card)
    ⊢ Exists fun F => And (HasSubset.Subset F A) (And (LE.le (↑F.card) K) (HasSubs …
  -/
  haveI : ∀ F, Decidable ((F : Set G).PairwiseDisjoint (· • B)) := fun F ↦ Classical.dec _
  /-
    G : Type u_1
    inst✝¹ : Group G
    K : Real
    inst✝ : DecidableEq G
    A B : Finset G
    hB : B.Nonempty
    hK : LE.le (↑(HMul.hMul A B).card) (HMul.hMul K ↑B.card)
    this : (F : Set G) → Decidable (F.PairwiseDisjoint fun x => HSMul.hSMul x B)
    ⊢ Exists fun F => And (HasSubset.Subset F A) (And (LE.le (↑F.card) K) (HasSubs …
  -/
  set C := {F ∈ A.powerset | F.toSet.PairwiseDisjoint (· • B)}
  obtain ⟨F, hF, hFmax⟩ := C.exists_maximal <| filter_nonempty_iff.2
    ⟨∅, empty_mem_powerset _, by rw [coe_empty]; exact Set.pairwiseDisjoint_empty⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    K : Real
    inst✝ : DecidableEq G
    A B : Finset G
    hB : B.Nonempty
    hK : LE.le (↑(HMul.hMul A B).card) (HMul.hMul K ↑B.card)
    this : (F : Set G) → Decidable (F.PairwiseDisjoint fun x => HSMul.hSMul x B)
    C : Finset (Finset G) := Finset.filter (fun F => (↑F).PairwiseDisjoint fun x = …
    F : Finset G
    hF : Membership.mem C F
    hFmax : ∀ (x : Finset G), Membership.mem C x → Not (LT.lt F x)
    ⊢ Exists fun F => And (HasSubset.Subset F A) (And (LE.le (↑F.card) K) (HasSubs …
  -/
  rw [mem_filter, mem_powerset] at hF
  /-
    case intro.intro
    G : Type u_1
    inst✝¹ : Group G
    K : Real
    inst✝ : DecidableEq G
    A B : Finset G
    hB : B.Nonempty
    hK : LE.le (↑(HMul.hMul A B).card) (HMul.hMul K ↑B.card)
    this : (F : Set G) → Decidable (F.PairwiseDisjoint fun x => HSMul.hSMul x B)
    C : Finset (Finset G) := Finset.filter (fun F => (↑F).PairwiseDisjoint fun x = …
    F : Finset G
    hF : And (HasSubset.Subset F A) ((↑F).PairwiseDisjoint fun x => HSMul.hSMul x B)
    hFmax : ∀ (x : Finset G), Membership.mem C x → Not (LT.lt F x)
    ⊢ Exists fun F => And (HasSubset.Subset F A) (And (LE.le (↑F.card) K) (HasSubs …
  -/
  obtain ⟨hFA, hF⟩ := hF
  /-
    case intro.intro.intro
    G : Type u_1
    inst✝¹ : Group G
    K : Real
    inst✝ : DecidableEq G
    A B : Finset G
    hB : B.Nonempty
    hK : LE.le (↑(HMul.hMul A B).card) (HMul.hMul K ↑B.card)
    this : (F : Set G) → Decidable (F.PairwiseDisjoint fun x => HSMul.hSMul x B)
    C : Finset (Finset G) := Finset.filter (fun F => (↑F).PairwiseDisjoint fun x = …
    F : Finset G
    hFmax : ∀ (x : Finset G), Membership.mem C x → Not (LT.lt F x)
    hFA : HasSubset.Subset F A
    hF : (↑F).PairwiseDisjoint fun x => HSMul.hSMul x B
    ⊢ Exists fun F => And (HasSubset.Subset F A) (And (LE.le (↑F.card) K) (HasSubs …
  -/
  refine ⟨F, hFA, le_of_mul_le_mul_right ?_ (by positivity : (0 : ℝ) < #B), fun a ha ↦ ?_⟩
  · calc
      (#F * #B : ℝ) = #(F * B) := by
        rw [card_mul_iff.2 <| pairwiseDisjoint_smul_iff.1 hF, Nat.cast_mul]
      _ ≤ #(A * B) := by gcongr
      _ ≤ K * #B := hK
  /-
    case intro.intro.intro.refine_2
    G : Type u_1
    inst✝¹ : Group G
    K : Real
    inst✝ : DecidableEq G
    A B : Finset G
    hB : B.Nonempty
    hK : LE.le (↑(HMul.hMul A B).card) (HMul.hMul K ↑B.card)
    this : (F : Set G) → Decidable (F.PairwiseDisjoint fun x => HSMul.hSMul x B)
    C : Finset (Finset G) := Finset.filter (fun F => (↑F).PairwiseDisjoint fun x = …
    F : Finset G
    hFmax : ∀ (x : Finset G), Membership.mem C x → Not (LT.lt F x)
    hFA : HasSubset.Subset F A
    hF : (↑F).PairwiseDisjoint fun x => HSMul.hSMul x B
    a : G
    ha : Membership.mem A a
    ⊢ Membership.mem (HMul.hMul F (HDiv.hDiv B B)) a
  -/
  by_cases hau : a ∈ F
    /-
      case pos
      G : Type u_1
      inst✝¹ : Group G
      K : Real
      inst✝ : DecidableEq G
      A B : Finset G
      hB : B.Nonempty
      hK : LE.le (↑(HMul.hMul A B).card) (HMul.hMul K ↑B.card)
      this : (F : Set G) → Decidable (F.PairwiseDisjoint fun x => HSMul.hSMul x B)
      C : Finset (Finset G) := Finset.filter (fun F => (↑F).PairwiseDisjoint fun x = …
      F : Finset G
      hFmax : ∀ (x : Finset G), Membership.mem C x → Not (LT.lt F x)
      hFA : HasSubset.Subset F A
      hF : (↑F).PairwiseDisjoint fun x => HSMul.hSMul x B
      a : G
      ha : Membership.mem A a
      hau : Membership.mem F a
      ⊢ Membership.mem (HMul.hMul F (HDiv.hDiv B B)) a
    -/
  · exact subset_mul_left _ hB.one_mem_div hau
    /-
      🎉 no goals
    -/
  /-
    case neg
    G : Type u_1
    inst✝¹ : Group G
    K : Real
    inst✝ : DecidableEq G
    A B : Finset G
    hB : B.Nonempty
    hK : LE.le (↑(HMul.hMul A B).card) (HMul.hMul K ↑B.card)
    this : (F : Set G) → Decidable (F.PairwiseDisjoint fun x => HSMul.hSMul x B)
    C : Finset (Finset G) := Finset.filter (fun F => (↑F).PairwiseDisjoint fun x = …
    F : Finset G
    hFmax : ∀ (x : Finset G), Membership.mem C x → Not (LT.lt F x)
    hFA : HasSubset.Subset F A
    hF : (↑F).PairwiseDisjoint fun x => HSMul.hSMul x B
    a : G
    ha : Membership.mem A a
    hau : Not (Membership.mem F a)
    ⊢ Membership.mem (HMul.hMul F (HDiv.hDiv B B)) a
  -/
  by_cases H : ∀ b ∈ F, Disjoint (a • B) (b • B)
    /-
      case pos
      G : Type u_1
      inst✝¹ : Group G
      K : Real
      inst✝ : DecidableEq G
      A B : Finset G
      hB : B.Nonempty
      hK : LE.le (↑(HMul.hMul A B).card) (HMul.hMul K ↑B.card)
      this : (F : Set G) → Decidable (F.PairwiseDisjoint fun x => HSMul.hSMul x B)
      C : Finset (Finset G) := Finset.filter (fun F => (↑F).PairwiseDisjoint fun x = …
      F : Finset G
      hFmax : ∀ (x : Finset G), Membership.mem C x → Not (LT.lt F x)
      hFA : HasSubset.Subset F A
      hF : (↑F).PairwiseDisjoint fun x => HSMul.hSMul x B
      a : G
      ha : Membership.mem A a
      hau : Not (Membership.mem F a)
      H : ∀ (b : G), Membership.mem F b → Disjoint (HSMul.hSMul a B) (HSMul.hSMul b B)
      ⊢ Membership.mem (HMul.hMul F (HDiv.hDiv B B)) a
    -/
  · refine (hFmax _ ?_ <| ssubset_insert hau).elim
    /-
      case pos
      G : Type u_1
      inst✝¹ : Group G
      K : Real
      inst✝ : DecidableEq G
      A B : Finset G
      hB : B.Nonempty
      hK : LE.le (↑(HMul.hMul A B).card) (HMul.hMul K ↑B.card)
      this : (F : Set G) → Decidable (F.PairwiseDisjoint fun x => HSMul.hSMul x B)
      C : Finset (Finset G) := Finset.filter (fun F => (↑F).PairwiseDisjoint fun x = …
      F : Finset G
      hFmax : ∀ (x : Finset G), Membership.mem C x → Not (LT.lt F x)
      hFA : HasSubset.Subset F A
      hF : (↑F).PairwiseDisjoint fun x => HSMul.hSMul x B
      a : G
      ha : Membership.mem A a
      hau : Not (Membership.mem F a)
      H : ∀ (b : G), Membership.mem F b → Disjoint (HSMul.hSMul a B) (HSMul.hSMul b B)
      ⊢ Membership.mem C (Insert.insert a F)
    -/
    rw [mem_filter, mem_powerset, insert_subset_iff, coe_insert]
    /-
      case pos
      G : Type u_1
      inst✝¹ : Group G
      K : Real
      inst✝ : DecidableEq G
      A B : Finset G
      hB : B.Nonempty
      hK : LE.le (↑(HMul.hMul A B).card) (HMul.hMul K ↑B.card)
      this : (F : Set G) → Decidable (F.PairwiseDisjoint fun x => HSMul.hSMul x B)
      C : Finset (Finset G) := Finset.filter (fun F => (↑F).PairwiseDisjoint fun x = …
      F : Finset G
      hFmax : ∀ (x : Finset G), Membership.mem C x → Not (LT.lt F x)
      hFA : HasSubset.Subset F A
      hF : (↑F).PairwiseDisjoint fun x => HSMul.hSMul x B
      a : G
      ha : Membership.mem A a
      hau : Not (Membership.mem F a)
      H : ∀ (b : G), Membership.mem F b → Disjoint (HSMul.hSMul a B) (HSMul.hSMul b B)
      ⊢ And (And (Membership.mem A a) (HasSubset.Subset F A)) ((Insert.insert a ↑F). …
    -/
    exact ⟨⟨ha, hFA⟩, hF.insert fun _ hb _ ↦ H _ hb⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    G : Type u_1
    inst✝¹ : Group G
    K : Real
    inst✝ : DecidableEq G
    A B : Finset G
    hB : B.Nonempty
    hK : LE.le (↑(HMul.hMul A B).card) (HMul.hMul K ↑B.card)
    this : (F : Set G) → Decidable (F.PairwiseDisjoint fun x => HSMul.hSMul x B)
    C : Finset (Finset G) := Finset.filter (fun F => (↑F).PairwiseDisjoint fun x = …
    F : Finset G
    hFmax : ∀ (x : Finset G), Membership.mem C x → Not (LT.lt F x)
    hFA : HasSubset.Subset F A
    hF : (↑F).PairwiseDisjoint fun x => HSMul.hSMul x B
    a : G
    ha : Membership.mem A a
    hau : Not (Membership.mem F a)
    H : Not (∀ (b : G), Membership.mem F b → Disjoint (HSMul.hSMul a B) (HSMul.hSM …
    ⊢ Membership.mem (HMul.hMul F (HDiv.hDiv B B)) a
  -/
  push_neg at H
  /-
    case neg
    G : Type u_1
    inst✝¹ : Group G
    K : Real
    inst✝ : DecidableEq G
    A B : Finset G
    hB : B.Nonempty
    hK : LE.le (↑(HMul.hMul A B).card) (HMul.hMul K ↑B.card)
    this : (F : Set G) → Decidable (F.PairwiseDisjoint fun x => HSMul.hSMul x B)
    C : Finset (Finset G) := Finset.filter (fun F => (↑F).PairwiseDisjoint fun x = …
    F : Finset G
    hFmax : ∀ (x : Finset G), Membership.mem C x → Not (LT.lt F x)
    hFA : HasSubset.Subset F A
    hF : (↑F).PairwiseDisjoint fun x => HSMul.hSMul x B
    a : G
    ha : Membership.mem A a
    hau : Not (Membership.mem F a)
    H : Exists fun b => And (Membership.mem F b) (Not (Disjoint (HSMul.hSMul a B)  …
    ⊢ Membership.mem (HMul.hMul F (HDiv.hDiv B B)) a
  -/
  simp_rw [not_disjoint_iff, ← inv_smul_mem_iff] at H
  /-
    case neg
    G : Type u_1
    inst✝¹ : Group G
    K : Real
    inst✝ : DecidableEq G
    A B : Finset G
    hB : B.Nonempty
    hK : LE.le (↑(HMul.hMul A B).card) (HMul.hMul K ↑B.card)
    this : (F : Set G) → Decidable (F.PairwiseDisjoint fun x => HSMul.hSMul x B)
    C : Finset (Finset G) := Finset.filter (fun F => (↑F).PairwiseDisjoint fun x = …
    F : Finset G
    hFmax : ∀ (x : Finset G), Membership.mem C x → Not (LT.lt F x)
    hFA : HasSubset.Subset F A
    hF : (↑F).PairwiseDisjoint fun x => HSMul.hSMul x B
    a : G
    ha : Membership.mem A a
    hau : Not (Membership.mem F a)
    H : Exists fun b => And (Membership.mem F b) (Exists fun a_1 => And (Membershi …
    ⊢ Membership.mem (HMul.hMul F (HDiv.hDiv B B)) a
  -/
  obtain ⟨b, hb, c, hc₁, hc₂⟩ := H
  /-
    case neg.intro.intro.intro.intro
    G : Type u_1
    inst✝¹ : Group G
    K : Real
    inst✝ : DecidableEq G
    A B : Finset G
    hB : B.Nonempty
    hK : LE.le (↑(HMul.hMul A B).card) (HMul.hMul K ↑B.card)
    this : (F : Set G) → Decidable (F.PairwiseDisjoint fun x => HSMul.hSMul x B)
    C : Finset (Finset G) := Finset.filter (fun F => (↑F).PairwiseDisjoint fun x = …
    F : Finset G
    hFmax : ∀ (x : Finset G), Membership.mem C x → Not (LT.lt F x)
    hFA : HasSubset.Subset F A
    hF : (↑F).PairwiseDisjoint fun x => HSMul.hSMul x B
    a : G
    ha : Membership.mem A a
    hau : Not (Membership.mem F a)
    b : G
    hb : Membership.mem F b
    c : G
    hc₁ : Membership.mem B (HSMul.hSMul (Inv.inv a) c)
    hc₂ : Membership.mem B (HSMul.hSMul (Inv.inv b) c)
    ⊢ Membership.mem (HMul.hMul F (HDiv.hDiv B B)) a
  -/
  exact mem_mul.2 ⟨b, hb, b⁻¹ * a, mem_div.2 ⟨_, hc₂, _, hc₁, by simp⟩, by simp⟩
  /-
    🎉 no goals
  -/

-- `alias` doesn't add the deprecation suggestion to the `to_additive` version
-- see https://github.com/leanprover-community/mathlib4/issues/19424

@[to_additive]
alias exists_subset_mul_div := ruzsa_covering_mul

/-- **Ruzsa's covering lemma** for sets. See also `Finset.ruzsa_covering_mul`. -/
@[to_additive "**Ruzsa's covering lemma** for sets. See also `Finset.ruzsa_covering_add`."]
lemma ruzsa_covering_mul (hA : A.Finite) (hB : B.Finite) (hB₀ : B.Nonempty)
    (hK : Nat.card (A * B) ≤ K * Nat.card B) :
    ∃ F ⊆ A, Nat.card F ≤ K ∧ A ⊆ F * (B / B) ∧ F.Finite := by
  /-
    G : Type u_1
    inst✝ : Group G
    K : Real
    A B : Set G
    hA : A.Finite
    hB : B.Finite
    hB₀ : B.Nonempty
    hK : LE.le (↑(Nat.card ↑(HMul.hMul A B))) (HMul.hMul K ↑(Nat.card ↑B))
    ⊢ Exists fun F => And (HasSubset.Subset F A) (And (LE.le (↑(Nat.card ↑F)) K) ( …
  -/
  lift A to Finset G using hA
  /-
    case intro
    G : Type u_1
    inst✝ : Group G
    K : Real
    B : Set G
    hB : B.Finite
    hB₀ : B.Nonempty
    A : Finset G
    hK : LE.le (↑(Nat.card ↑(HMul.hMul (↑A) B))) (HMul.hMul K ↑(Nat.card ↑B))
    ⊢ Exists fun F => And (HasSubset.Subset F ↑A) (And (LE.le (↑(Nat.card ↑F)) K)  …
  -/
  lift B to Finset G using hB
  classical
  obtain ⟨F, hFA, hF, hAF⟩ := Finset.ruzsa_covering_mul hB₀ (by simpa [← Finset.coe_mul] using hK)
  exact ⟨F, by norm_cast; simp [*]⟩

-- `alias` doesn't add the deprecation suggestion to the `to_additive` version
-- see https://github.com/leanprover-community/mathlib4/issues/19424

