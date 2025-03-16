protected theorem IsGδ.setOf_irrational : IsGδ { x | Irrational x } :=
  (countable_range _).isGδ_compl


@[deprecated (since := "2024-02-15")] alias isGδ_irrational := IsGδ.setOf_irrational


theorem dense_irrational : Dense { x : ℝ | Irrational x } := by
  /-
    ⊢ Dense (setOf fun x => Irrational x)
  -/
  refine Real.isTopologicalBasis_Ioo_rat.dense_iff.2 ?_
  /-
    ⊢ ∀ (o : Set Real), Membership.mem (Set.iUnion fun a => Set.iUnion fun b => Se …
  -/
  simp only [mem_iUnion, mem_singleton_iff, exists_prop, forall_exists_index, and_imp]
  /-
    ⊢ ∀ (o : Set Real) (x x_1 : Rat), LT.lt x x_1 → Eq o (Set.Ioo ↑x ↑x_1) → o.Non …
  -/
  rintro _ a b hlt rfl _
  /-
    a b : Rat
    hlt : LT.lt a b
    a✝ : (Set.Ioo ↑a ↑b).Nonempty
    ⊢ (Inter.inter (Set.Ioo ↑a ↑b) (setOf fun x => Irrational x)).Nonempty
  -/
  rw [inter_comm]
  /-
    a b : Rat
    hlt : LT.lt a b
    a✝ : (Set.Ioo ↑a ↑b).Nonempty
    ⊢ (Inter.inter (setOf fun x => Irrational x) (Set.Ioo ↑a ↑b)).Nonempty
  -/
  exact exists_irrational_btwn (Rat.cast_lt.2 hlt)
  /-
    🎉 no goals
  -/


theorem eventually_residual_irrational : ∀ᶠ x in residual ℝ, Irrational x :=
  residual_of_dense_Gδ .setOf_irrational dense_irrational


instance : OrderTopology { x // Irrational x } :=
  induced_orderTopology _ Iff.rfl <| @fun _ _ hlt =>
    let ⟨z, hz, hxz, hzy⟩ := exists_irrational_btwn hlt
    ⟨⟨z, hz⟩, hxz, hzy⟩


instance : NoMaxOrder { x // Irrational x } :=
                                                   /-
                                                     x✝¹ : Real
                                                     x✝ : Subtype fun x => Irrational x
                                                     x : Real
                                                     hx : Irrational x
                                                     ⊢ LT.lt ⟨x, hx⟩ ⟨HAdd.hAdd x ↑1, ⋯⟩
                                                   -/
  ⟨fun ⟨x, hx⟩ => ⟨⟨x + (1 : ℕ), hx.add_nat 1⟩, by simp⟩⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


instance : NoMinOrder { x // Irrational x } :=
                                                   /-
                                                     x✝¹ : Real
                                                     x✝ : Subtype fun x => Irrational x
                                                     x : Real
                                                     hx : Irrational x
                                                     ⊢ LT.lt ⟨HSub.hSub x ↑1, ⋯⟩ ⟨x, hx⟩
                                                   -/
  ⟨fun ⟨x, hx⟩ => ⟨⟨x - (1 : ℕ), hx.sub_nat 1⟩, by simp⟩⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


instance : DenselyOrdered { x // Irrational x } :=
  ⟨fun _ _ hlt =>
    let ⟨z, hz, hxz, hzy⟩ := exists_irrational_btwn hlt
    ⟨⟨z, hz⟩, hxz, hzy⟩⟩


theorem eventually_forall_le_dist_cast_div (hx : Irrational x) (n : ℕ) :
    ∀ᶠ ε : ℝ in 𝓝 0, ∀ m : ℤ, ε ≤ dist x (m / n) := by
  have A : IsClosed (range (fun m => (n : ℝ)⁻¹ * m : ℤ → ℝ)) :=
    ((isClosedMap_smul₀ (n⁻¹ : ℝ)).comp Int.isClosedEmbedding_coe_real.isClosedMap).isClosed_range
  have B : x ∉ range (fun m => (n : ℝ)⁻¹ * m : ℤ → ℝ) := by
    rintro ⟨m, rfl⟩
    simp at hx
  /-
    x : Real
    hx : Irrational x
    n : Nat
    A : IsClosed (Set.range fun m => HMul.hMul (Inv.inv ↑n) ↑m)
    B : Not (Membership.mem (Set.range fun m => HMul.hMul (Inv.inv ↑n) ↑m) x)
    ⊢ Filter.Eventually (fun ε => ∀ (m : Int), LE.le ε (Dist.dist x (HDiv.hDiv ↑m  …
  -/
  rcases Metric.mem_nhds_iff.1 (A.isOpen_compl.mem_nhds B) with ⟨ε, ε0, hε⟩
  /-
    case intro.intro
    x : Real
    hx : Irrational x
    n : Nat
    A : IsClosed (Set.range fun m => HMul.hMul (Inv.inv ↑n) ↑m)
    B : Not (Membership.mem (Set.range fun m => HMul.hMul (Inv.inv ↑n) ↑m) x)
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball x ε) (HasCompl.compl (Set.range fun m => HM …
    ⊢ Filter.Eventually (fun ε => ∀ (m : Int), LE.le ε (Dist.dist x (HDiv.hDiv ↑m  …
  -/
  refine (ge_mem_nhds ε0).mono fun δ hδ m => not_lt.1 fun hlt => ?_
  /-
    case intro.intro
    x : Real
    hx : Irrational x
    n : Nat
    A : IsClosed (Set.range fun m => HMul.hMul (Inv.inv ↑n) ↑m)
    B : Not (Membership.mem (Set.range fun m => HMul.hMul (Inv.inv ↑n) ↑m) x)
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball x ε) (HasCompl.compl (Set.range fun m => HM …
    δ : Real
    hδ : LE.le δ ε
    m : Int
    hlt : LT.lt (Dist.dist x (HDiv.hDiv ↑m ↑n)) δ
    ⊢ False
  -/
  rw [dist_comm] at hlt
  /-
    case intro.intro
    x : Real
    hx : Irrational x
    n : Nat
    A : IsClosed (Set.range fun m => HMul.hMul (Inv.inv ↑n) ↑m)
    B : Not (Membership.mem (Set.range fun m => HMul.hMul (Inv.inv ↑n) ↑m) x)
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball x ε) (HasCompl.compl (Set.range fun m => HM …
    δ : Real
    hδ : LE.le δ ε
    m : Int
    hlt : LT.lt (Dist.dist (HDiv.hDiv ↑m ↑n) x) δ
    ⊢ False
  -/
  refine hε (ball_subset_ball hδ hlt) ⟨m, ?_⟩
  /-
    case intro.intro
    x : Real
    hx : Irrational x
    n : Nat
    A : IsClosed (Set.range fun m => HMul.hMul (Inv.inv ↑n) ↑m)
    B : Not (Membership.mem (Set.range fun m => HMul.hMul (Inv.inv ↑n) ↑m) x)
    ε : Real
    ε0 : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball x ε) (HasCompl.compl (Set.range fun m => HM …
    δ : Real
    hδ : LE.le δ ε
    m : Int
    hlt : LT.lt (Dist.dist (HDiv.hDiv ↑m ↑n) x) δ
    ⊢ Eq ((fun m => HMul.hMul (Inv.inv ↑n) ↑m) m) (HDiv.hDiv ↑m ↑n)
  -/
  simp [div_eq_inv_mul]
  /-
    🎉 no goals
  -/


theorem eventually_forall_le_dist_cast_div_of_denom_le (hx : Irrational x) (n : ℕ) :
    ∀ᶠ ε : ℝ in 𝓝 0, ∀ k ≤ n, ∀ (m : ℤ), ε ≤ dist x (m / k) :=
  (finite_le_nat n).eventually_all.2 fun k _ => hx.eventually_forall_le_dist_cast_div k


theorem eventually_forall_le_dist_cast_rat_of_den_le (hx : Irrational x) (n : ℕ) :
    ∀ᶠ ε : ℝ in 𝓝 0, ∀ r : ℚ, r.den ≤ n → ε ≤ dist x r :=
  (hx.eventually_forall_le_dist_cast_div_of_denom_le n).mono fun ε H r hr => by
    /-
      x : Real
      hx : Irrational x
      n : Nat
      ε : Real
      H : ∀ (k : Nat), LE.le k n → ∀ (m : Int), LE.le ε (Dist.dist x (HDiv.hDiv ↑m ↑ …
      r : Rat
      hr : LE.le r.den n
      ⊢ LE.le ε (Dist.dist x ↑r)
    -/
    simpa only [Rat.cast_def] using H r.den hr r.num
    /-
      🎉 no goals
    -/


