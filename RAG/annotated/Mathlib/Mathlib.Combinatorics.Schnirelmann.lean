/-- The Schnirelmann density is defined as the infimum of |A ∩ {1, ..., n}| / n as n ranges over
the positive naturals. -/
noncomputable def schnirelmannDensity (A : Set ℕ) [DecidablePred (· ∈ A)] : ℝ :=
  ⨅ n : {n : ℕ // 0 < n}, #{a ∈ Ioc 0 n | a ∈ A} / n


lemma schnirelmannDensity_nonneg : 0 ≤ schnirelmannDensity A :=
                                /-
                                  A : Set Nat
                                  inst✝ : DecidablePred fun x => Membership.mem A x
                                  x✝ : Subtype fun n => LT.lt 0 n
                                  ⊢ LE.le 0 (HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem A a) (Finset.Ioc …
                                -/
  Real.iInf_nonneg (fun _ => by positivity)
                                /-
                                  🎉 no goals
                                -/


lemma schnirelmannDensity_le_div {n : ℕ} (hn : n ≠ 0) :
    schnirelmannDensity A ≤ #{a ∈ Ioc 0 n | a ∈ A} / n :=
                                        /-
                                          A : Set Nat
                                          inst✝ : DecidablePred fun x => Membership.mem A x
                                          n : Nat
                                          hn : Ne n 0
                                          x✝¹ : Real
                                          x✝ : Membership.mem (Set.range fun n => HDiv.hDiv ↑(Finset.filter (fun a => Me …
                                          w✝ : Subtype fun n => LT.lt 0 n
                                          hx : Eq ((fun n => HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem A a) (Fi …
                                          ⊢ LE.le 0 ((fun n => HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem A a) ( …
                                        -/
  ciInf_le ⟨0, fun _ ⟨_, hx⟩ => hx ▸ by positivity⟩ (⟨n, hn.bot_lt⟩ : {n : ℕ // 0 < n})
                                        /-
                                          🎉 no goals
                                        -/


/--
For any natural `n`, the Schnirelmann density multiplied by `n` is bounded by `|A ∩ {1, ..., n}|`.
Note this property fails for the natural density.
-/
lemma schnirelmannDensity_mul_le_card_filter {n : ℕ} :
    schnirelmannDensity A * n ≤ #{a ∈ Ioc 0 n | a ∈ A} := by
  /-
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    n : Nat
    ⊢ LE.le (HMul.hMul (schnirelmannDensity A) ↑n) ↑(Finset.filter (fun a => Membe …
  -/
  rcases eq_or_ne n 0 with rfl | hn
    /-
      case inl
      A : Set Nat
      inst✝ : DecidablePred fun x => Membership.mem A x
      ⊢ LE.le (HMul.hMul (schnirelmannDensity A) ↑0) ↑(Finset.filter (fun a => Membe …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    n : Nat
    hn : Ne n 0
    ⊢ LE.le (HMul.hMul (schnirelmannDensity A) ↑n) ↑(Finset.filter (fun a => Membe …
  -/
  exact (le_div_iff₀ (by positivity)).1 (schnirelmannDensity_le_div hn)
  /-
    🎉 no goals
  -/


/--
To show the Schnirelmann density is upper bounded by `x`, it suffices to show
`|A ∩ {1, ..., n}| / n ≤ x`, for any chosen positive value of `n`.

We provide `n` explicitly here to make this lemma more easily usable in `apply` or `refine`.
This lemma is analogous to `ciInf_le_of_le`.
-/
lemma schnirelmannDensity_le_of_le {x : ℝ} (n : ℕ) (hn : n ≠ 0)
    (hx : #{a ∈ Ioc 0 n | a ∈ A} / n ≤ x) : schnirelmannDensity A ≤ x :=
  (schnirelmannDensity_le_div hn).trans hx


lemma schnirelmannDensity_le_one : schnirelmannDensity A ≤ 1 :=
  schnirelmannDensity_le_of_le 1 one_ne_zero <|
       /-
         A : Set Nat
         inst✝ : DecidablePred fun x => Membership.mem A x
         ⊢ LE.le (HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem A a) (Finset.Ioc 0 …
       -/
    by rw [Nat.cast_one, div_one, Nat.cast_le_one]; exact card_filter_le _ _
                                                    /-
                                                      🎉 no goals
                                                    -/


/--
If `k` is omitted from the set, its Schnirelmann density is upper bounded by `1 - k⁻¹`.
-/
lemma schnirelmannDensity_le_of_not_mem {k : ℕ} (hk : k ∉ A) :
    schnirelmannDensity A ≤ 1 - (k⁻¹ : ℝ) := by
  /-
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    k : Nat
    hk : Not (Membership.mem A k)
    ⊢ LE.le (schnirelmannDensity A) (HSub.hSub 1 (Inv.inv ↑k))
  -/
  rcases k.eq_zero_or_pos with rfl | hk'
    /-
      case inl
      A : Set Nat
      inst✝ : DecidablePred fun x => Membership.mem A x
      hk : Not (Membership.mem A 0)
      ⊢ LE.le (schnirelmannDensity A) (HSub.hSub 1 (Inv.inv ↑0))
    -/
  · simpa using schnirelmannDensity_le_one
    /-
      🎉 no goals
    -/
  /-
    case inr
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    k : Nat
    hk : Not (Membership.mem A k)
    hk' : GT.gt k 0
    ⊢ LE.le (schnirelmannDensity A) (HSub.hSub 1 (Inv.inv ↑k))
  -/
  apply schnirelmannDensity_le_of_le k hk'.ne'
  /-
    case inr
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    k : Nat
    hk : Not (Membership.mem A k)
    hk' : GT.gt k 0
    ⊢ LE.le (HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem A a) (Finset.Ioc 0 …
  -/
  rw [← one_div, one_sub_div (Nat.cast_pos.2 hk').ne']
  /-
    case inr
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    k : Nat
    hk : Not (Membership.mem A k)
    hk' : GT.gt k 0
    ⊢ LE.le (HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem A a) (Finset.Ioc 0 …
  -/
  gcongr
  /-
    case inr.hab
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    k : Nat
    hk : Not (Membership.mem A k)
    hk' : GT.gt k 0
    ⊢ LE.le (↑(Finset.filter (fun a => Membership.mem A a) (Finset.Ioc 0 k)).card) …
  -/
  rw [← Nat.cast_pred hk', Nat.cast_le]
  /-
    case inr.hab
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    k : Nat
    hk : Not (Membership.mem A k)
    hk' : GT.gt k 0
    ⊢ LE.le (Finset.filter (fun a => Membership.mem A a) (Finset.Ioc 0 k)).card (H …
  -/
  suffices {a ∈ Ioc 0 k | a ∈ A} ⊆ Ioo 0 k from (card_le_card this).trans_eq (by simp)
  /-
    case inr.hab
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    k : Nat
    hk : Not (Membership.mem A k)
    hk' : GT.gt k 0
    ⊢ HasSubset.Subset (Finset.filter (fun a => Membership.mem A a) (Finset.Ioc 0  …
  -/
  rw [← Ioo_insert_right hk', filter_insert, if_neg hk]
  /-
    case inr.hab
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    k : Nat
    hk : Not (Membership.mem A k)
    hk' : GT.gt k 0
    ⊢ HasSubset.Subset (Finset.filter (fun a => Membership.mem A a) (Finset.Ioo 0  …
  -/
  exact filter_subset _ _
  /-
    🎉 no goals
  -/


/-- The Schnirelmann density of a set not containing `1` is `0`. -/
lemma schnirelmannDensity_eq_zero_of_one_not_mem (h : 1 ∉ A) : schnirelmannDensity A = 0 :=
                                                   /-
                                                     A : Set Nat
                                                     inst✝ : DecidablePred fun x => Membership.mem A x
                                                     h : Not (Membership.mem A 1)
                                                     ⊢ LE.le (HSub.hSub 1 (Inv.inv ↑1)) 0
                                                   -/
  ((schnirelmannDensity_le_of_not_mem h).trans (by simp)).antisymm schnirelmannDensity_nonneg
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- The Schnirelmann density is increasing with the set. -/
lemma schnirelmannDensity_le_of_subset {B : Set ℕ} [DecidablePred (· ∈ B)] (h : A ⊆ B) :
    schnirelmannDensity A ≤ schnirelmannDensity B :=
                                         /-
                                           A : Set Nat
                                           inst✝¹ : DecidablePred fun x => Membership.mem A x
                                           B : Set Nat
                                           inst✝ : DecidablePred fun x => Membership.mem B x
                                           h : HasSubset.Subset A B
                                           x✝¹ : Real
                                           x✝ : Membership.mem (Set.range fun n => HDiv.hDiv ↑(Finset.filter (fun a => Me …
                                           w✝ : Subtype fun n => LT.lt 0 n
                                           hx : Eq ((fun n => HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem A a) (Fi …
                                           ⊢ LE.le 0 ((fun n => HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem A a) ( …
                                         -/
  ciInf_mono ⟨0, fun _ ⟨_, hx⟩ ↦ hx ▸ by positivity⟩ fun _ ↦ by
                                         /-
                                           🎉 no goals
                                         -/
    /-
      A : Set Nat
      inst✝¹ : DecidablePred fun x => Membership.mem A x
      B : Set Nat
      inst✝ : DecidablePred fun x => Membership.mem B x
      h : HasSubset.Subset A B
      x✝ : Subtype fun n => LT.lt 0 n
      ⊢ LE.le (HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem A a) (Finset.Ioc 0 …
    -/
    gcongr; exact h
            /-
              🎉 no goals
            -/


/-- The Schnirelmann density of `A` is `1` if and only if `A` contains all the positive naturals. -/
lemma schnirelmannDensity_eq_one_iff : schnirelmannDensity A = 1 ↔ {0}ᶜ ⊆ A := by
  /-
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    ⊢ Iff (Eq (schnirelmannDensity A) 1) (HasSubset.Subset (HasCompl.compl (Single …
  -/
  rw [le_antisymm_iff, and_iff_right schnirelmannDensity_le_one]
  /-
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    ⊢ Iff (LE.le 1 (schnirelmannDensity A)) (HasSubset.Subset (HasCompl.compl (Sin …
  -/
  constructor
    /-
      case mp
      A : Set Nat
      inst✝ : DecidablePred fun x => Membership.mem A x
      ⊢ LE.le 1 (schnirelmannDensity A) → HasSubset.Subset (HasCompl.compl (Singleto …
    -/
  · rw [← not_imp_not, not_le]
    /-
      case mp
      A : Set Nat
      inst✝ : DecidablePred fun x => Membership.mem A x
      ⊢ Not (HasSubset.Subset (HasCompl.compl (Singleton.singleton 0)) A) → LT.lt (s …
    -/
    simp only [Set.not_subset, forall_exists_index, true_and, and_imp, Set.mem_singleton_iff]
    /-
      case mp
      A : Set Nat
      inst✝ : DecidablePred fun x => Membership.mem A x
      ⊢ ∀ (x : Nat), Membership.mem (HasCompl.compl (Singleton.singleton 0)) x → Not …
    -/
    intro x hx hx'
    /-
      case mp
      A : Set Nat
      inst✝ : DecidablePred fun x => Membership.mem A x
      x : Nat
      hx : Membership.mem (HasCompl.compl (Singleton.singleton 0)) x
      hx' : Not (Membership.mem A x)
      ⊢ LT.lt (schnirelmannDensity A) 1
    -/
    apply (schnirelmannDensity_le_of_not_mem hx').trans_lt
    /-
      case mp
      A : Set Nat
      inst✝ : DecidablePred fun x => Membership.mem A x
      x : Nat
      hx : Membership.mem (HasCompl.compl (Singleton.singleton 0)) x
      hx' : Not (Membership.mem A x)
      ⊢ LT.lt (HSub.hSub 1 (Inv.inv ↑x)) 1
    -/
    simpa only [one_div, sub_lt_self_iff, inv_pos, Nat.cast_pos, pos_iff_ne_zero] using hx
    /-
      🎉 no goals
    -/
    /-
      case mpr
      A : Set Nat
      inst✝ : DecidablePred fun x => Membership.mem A x
      ⊢ HasSubset.Subset (HasCompl.compl (Singleton.singleton 0)) A → LE.le 1 (schni …
    -/
  · intro h
    /-
      case mpr
      A : Set Nat
      inst✝ : DecidablePred fun x => Membership.mem A x
      h : HasSubset.Subset (HasCompl.compl (Singleton.singleton 0)) A
      ⊢ LE.le 1 (schnirelmannDensity A)
    -/
    refine le_ciInf fun ⟨n, hn⟩ => ?_
    /-
      case mpr
      A : Set Nat
      inst✝ : DecidablePred fun x => Membership.mem A x
      h : HasSubset.Subset (HasCompl.compl (Singleton.singleton 0)) A
      x✝ : Subtype fun n => LT.lt 0 n
      n : Nat
      hn : LT.lt 0 n
      ⊢ LE.le 1 (HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem A a) (Finset.Ioc …
    -/
    rw [one_le_div (Nat.cast_pos.2 hn), Nat.cast_le, filter_true_of_mem, Nat.card_Ioc, Nat.sub_zero]
    /-
      case mpr
      A : Set Nat
      inst✝ : DecidablePred fun x => Membership.mem A x
      h : HasSubset.Subset (HasCompl.compl (Singleton.singleton 0)) A
      x✝ : Subtype fun n => LT.lt 0 n
      n : Nat
      hn : LT.lt 0 n
      ⊢ ∀ (x : Nat), Membership.mem (Finset.Ioc 0 ↑⟨n, hn⟩) x → Membership.mem A x
    -/
    rintro x hx
    /-
      case mpr
      A : Set Nat
      inst✝ : DecidablePred fun x => Membership.mem A x
      h : HasSubset.Subset (HasCompl.compl (Singleton.singleton 0)) A
      x✝ : Subtype fun n => LT.lt 0 n
      n : Nat
      hn : LT.lt 0 n
      x : Nat
      hx : Membership.mem (Finset.Ioc 0 ↑⟨n, hn⟩) x
      ⊢ Membership.mem A x
    -/
    exact h (mem_Ioc.1 hx).1.ne'
    /-
      🎉 no goals
    -/


/-- The Schnirelmann density of `A` containing `0` is `1` if and only if `A` is the naturals. -/
lemma schnirelmannDensity_eq_one_iff_of_zero_mem (hA : 0 ∈ A) :
    schnirelmannDensity A = 1 ↔ A = Set.univ := by
  /-
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    hA : Membership.mem A 0
    ⊢ Iff (Eq (schnirelmannDensity A) 1) (Eq A Set.univ)
  -/
  rw [schnirelmannDensity_eq_one_iff]
  /-
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    hA : Membership.mem A 0
    ⊢ Iff (HasSubset.Subset (HasCompl.compl (Singleton.singleton 0)) A) (Eq A Set. …
  -/
  constructor
    /-
      case mp
      A : Set Nat
      inst✝ : DecidablePred fun x => Membership.mem A x
      hA : Membership.mem A 0
      ⊢ HasSubset.Subset (HasCompl.compl (Singleton.singleton 0)) A → Eq A Set.univ
    -/
  · refine fun h => Set.eq_univ_of_forall fun x => ?_
    /-
      case mp
      A : Set Nat
      inst✝ : DecidablePred fun x => Membership.mem A x
      hA : Membership.mem A 0
      h : HasSubset.Subset (HasCompl.compl (Singleton.singleton 0)) A
      x : Nat
      ⊢ Membership.mem A x
    -/
    rcases eq_or_ne x 0 with rfl | hx
      /-
        case mp.inl
        A : Set Nat
        inst✝ : DecidablePred fun x => Membership.mem A x
        hA : Membership.mem A 0
        h : HasSubset.Subset (HasCompl.compl (Singleton.singleton 0)) A
        ⊢ Membership.mem A 0
      -/
    · exact hA
      /-
        🎉 no goals
      -/
      /-
        case mp.inr
        A : Set Nat
        inst✝ : DecidablePred fun x => Membership.mem A x
        hA : Membership.mem A 0
        h : HasSubset.Subset (HasCompl.compl (Singleton.singleton 0)) A
        x : Nat
        hx : Ne x 0
        ⊢ Membership.mem A x
      -/
    · exact h hx
      /-
        🎉 no goals
      -/
    /-
      case mpr
      A : Set Nat
      inst✝ : DecidablePred fun x => Membership.mem A x
      hA : Membership.mem A 0
      ⊢ Eq A Set.univ → HasSubset.Subset (HasCompl.compl (Singleton.singleton 0)) A
    -/
  · rintro rfl
    /-
      case mpr
      inst✝ : DecidablePred fun x => Membership.mem Set.univ x
      hA : Membership.mem Set.univ 0
      ⊢ HasSubset.Subset (HasCompl.compl (Singleton.singleton 0)) Set.univ
    -/
    exact Set.subset_univ {0}ᶜ
    /-
      🎉 no goals
    -/


lemma le_schnirelmannDensity_iff {x : ℝ} :
    x ≤ schnirelmannDensity A ↔ ∀ n : ℕ, 0 < n → x ≤ #{a ∈ Ioc 0 n | a ∈ A} / n :=
                                             /-
                                               A : Set Nat
                                               inst✝ : DecidablePred fun x => Membership.mem A x
                                               x x✝¹ : Real
                                               x✝ : Membership.mem (Set.range fun n => HDiv.hDiv ↑(Finset.filter (fun a => Me …
                                               w✝ : Subtype fun n => LT.lt 0 n
                                               hx : Eq ((fun n => HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem A a) (Fi …
                                               ⊢ LE.le 0 ((fun n => HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem A a) ( …
                                             -/
  (le_ciInf_iff ⟨0, fun _ ⟨_, hx⟩ => hx ▸ by positivity⟩).trans Subtype.forall
                                             /-
                                               🎉 no goals
                                             -/


lemma schnirelmannDensity_lt_iff {x : ℝ} :
    schnirelmannDensity A < x ↔ ∃ n : ℕ, 0 < n ∧ #{a ∈ Ioc 0 n | a ∈ A} / n < x := by
  /-
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    x : Real
    ⊢ Iff (LT.lt (schnirelmannDensity A) x) (Exists fun n => And (LT.lt 0 n) (LT.l …
  -/
  rw [← not_le, le_schnirelmannDensity_iff]; simp
                                             /-
                                               🎉 no goals
                                             -/


lemma schnirelmannDensity_le_iff_forall {x : ℝ} :
    schnirelmannDensity A ≤ x ↔
      ∀ ε : ℝ, 0 < ε → ∃ n : ℕ, 0 < n ∧ #{a ∈ Ioc 0 n | a ∈ A} / n < x + ε := by
  /-
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    x : Real
    ⊢ Iff (LE.le (schnirelmannDensity A) x) (∀ (ε : Real), LT.lt 0 ε → Exists fun  …
  -/
  rw [le_iff_forall_pos_lt_add]
  /-
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    x : Real
    ⊢ Iff (∀ (ε : Real), LT.lt 0 ε → LT.lt (schnirelmannDensity A) (HAdd.hAdd x ε) …
  -/
  simp only [schnirelmannDensity_lt_iff]
  /-
    🎉 no goals
  -/


lemma schnirelmannDensity_congr' {B : Set ℕ} [DecidablePred (· ∈ B)]
    (h : ∀ n > 0, n ∈ A ↔ n ∈ B) : schnirelmannDensity A = schnirelmannDensity B := by
  /-
    A : Set Nat
    inst✝¹ : DecidablePred fun x => Membership.mem A x
    B : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem B x
    h : ∀ (n : Nat), GT.gt n 0 → Iff (Membership.mem A n) (Membership.mem B n)
    ⊢ Eq (schnirelmannDensity A) (schnirelmannDensity B)
  -/
  rw [schnirelmannDensity, schnirelmannDensity]; congr; ext ⟨n, hn⟩; congr 3; ext x; aesop
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


/-- The Schnirelmann density is unaffected by adding `0`. -/
@[simp] lemma schnirelmannDensity_insert_zero [DecidablePred (· ∈ insert 0 A)] :
    schnirelmannDensity (insert 0 A) = schnirelmannDensity A :=
                                 /-
                                   A : Set Nat
                                   inst✝¹ : DecidablePred fun x => Membership.mem A x
                                   inst✝ : DecidablePred fun x => Membership.mem (Insert.insert 0 A) x
                                   ⊢ ∀ (n : Nat), GT.gt n 0 → Iff (Membership.mem (Insert.insert 0 A) n) (Members …
                                 -/
  schnirelmannDensity_congr' (by aesop)
                                 /-
                                   🎉 no goals
                                 -/


/-- The Schnirelmann density is unaffected by removing `0`. -/
lemma schnirelmannDensity_diff_singleton_zero [DecidablePred (· ∈ A \ {0})] :
    schnirelmannDensity (A \ {0}) = schnirelmannDensity A :=
                                 /-
                                   A : Set Nat
                                   inst✝¹ : DecidablePred fun x => Membership.mem A x
                                   inst✝ : DecidablePred fun x => Membership.mem (SDiff.sdiff A (Singleton.single …
                                   ⊢ ∀ (n : Nat), GT.gt n 0 → Iff (Membership.mem (SDiff.sdiff A (Singleton.singl …
                                 -/
  schnirelmannDensity_congr' (by aesop)
                                 /-
                                   🎉 no goals
                                 -/


lemma schnirelmannDensity_congr {B : Set ℕ} [DecidablePred (· ∈ B)] (h : A = B) :
    schnirelmannDensity A = schnirelmannDensity B :=
                                 /-
                                   A : Set Nat
                                   inst✝¹ : DecidablePred fun x => Membership.mem A x
                                   B : Set Nat
                                   inst✝ : DecidablePred fun x => Membership.mem B x
                                   h : Eq A B
                                   ⊢ ∀ (n : Nat), GT.gt n 0 → Iff (Membership.mem A n) (Membership.mem B n)
                                 -/
  schnirelmannDensity_congr' (by aesop)
                                 /-
                                   🎉 no goals
                                 -/


/--
If the Schnirelmann density is `0`, there is a positive natural for which
`|A ∩ {1, ..., n}| / n < ε`, for any positive `ε`.
Note this cannot be improved to `∃ᶠ n : ℕ in atTop`, as can be seen by `A = {1}ᶜ`.
-/
lemma exists_of_schnirelmannDensity_eq_zero {ε : ℝ} (hε : 0 < ε) (hA : schnirelmannDensity A = 0) :
    ∃ n, 0 < n ∧ #{a ∈ Ioc 0 n | a ∈ A} / n < ε := by
  /-
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    ε : Real
    hε : LT.lt 0 ε
    hA : Eq (schnirelmannDensity A) 0
    ⊢ Exists fun n => And (LT.lt 0 n) (LT.lt (HDiv.hDiv ↑(Finset.filter (fun a =>  …
  -/
  by_contra! h
  /-
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    ε : Real
    hε : LT.lt 0 ε
    hA : Eq (schnirelmannDensity A) 0
    h : ∀ (n : Nat), LT.lt 0 n → LE.le ε (HDiv.hDiv ↑(Finset.filter (fun a => Memb …
    ⊢ False
  -/
  rw [← le_schnirelmannDensity_iff] at h
  /-
    A : Set Nat
    inst✝ : DecidablePred fun x => Membership.mem A x
    ε : Real
    hε : LT.lt 0 ε
    hA : Eq (schnirelmannDensity A) 0
    h : LE.le ε (schnirelmannDensity A)
    ⊢ False
  -/
  linarith
  /-
    🎉 no goals
  -/


@[simp] lemma schnirelmannDensity_empty : schnirelmannDensity ∅ = 0 :=
                                                 /-
                                                   ⊢ Not (Membership.mem EmptyCollection.emptyCollection 1)
                                                 -/
  schnirelmannDensity_eq_zero_of_one_not_mem (by simp)
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- The Schnirelmann density of any finset is `0`. -/
lemma schnirelmannDensity_finset (A : Finset ℕ) : schnirelmannDensity A = 0 := by
  /-
    A : Finset Nat
    ⊢ Eq (schnirelmannDensity ↑A) 0
  -/
  refine le_antisymm ?_ schnirelmannDensity_nonneg
  /-
    A : Finset Nat
    ⊢ LE.le (schnirelmannDensity ↑A) 0
  -/
  simp only [schnirelmannDensity_le_iff_forall, zero_add]
  /-
    A : Finset Nat
    ⊢ ∀ (ε : Real), LT.lt 0 ε → Exists fun n => And (LT.lt 0 n) (LT.lt (HDiv.hDiv  …
  -/
  intro ε hε
  /-
    A : Finset Nat
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun n => And (LT.lt 0 n) (LT.lt (HDiv.hDiv ↑(Finset.filter (fun a =>  …
  -/
  wlog hε₁ : ε ≤ 1 generalizing ε
    /-
      case inr
      A : Finset Nat
      ε : Real
      hε : LT.lt 0 ε
      this : ∀ (ε : Real), LT.lt 0 ε → LE.le ε 1 → Exists fun n => And (LT.lt 0 n) ( …
      hε₁ : Not (LE.le ε 1)
      ⊢ Exists fun n => And (LT.lt 0 n) (LT.lt (HDiv.hDiv ↑(Finset.filter (fun a =>  …
    -/
  · obtain ⟨n, hn, hn'⟩ := this 1 zero_lt_one le_rfl
    /-
      case inr.intro.intro
      A : Finset Nat
      ε : Real
      hε : LT.lt 0 ε
      this : ∀ (ε : Real), LT.lt 0 ε → LE.le ε 1 → Exists fun n => And (LT.lt 0 n) ( …
      hε₁ : Not (LE.le ε 1)
      n : Nat
      hn : LT.lt 0 n
      hn' : LT.lt (HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem (↑A) a) (Finse …
      ⊢ Exists fun n => And (LT.lt 0 n) (LT.lt (HDiv.hDiv ↑(Finset.filter (fun a =>  …
    -/
    exact ⟨n, hn, hn'.trans_le (le_of_not_le hε₁)⟩
    /-
      🎉 no goals
    -/
  /-
    A : Finset Nat
    ε : Real
    hε : LT.lt 0 ε
    hε₁ : LE.le ε 1
    ⊢ Exists fun n => And (LT.lt 0 n) (LT.lt (HDiv.hDiv ↑(Finset.filter (fun a =>  …
  -/
  let n : ℕ := ⌊#A / ε⌋₊ + 1
  /-
    A : Finset Nat
    ε : Real
    hε : LT.lt 0 ε
    hε₁ : LE.le ε 1
    n : Nat := HAdd.hAdd (Nat.floor (HDiv.hDiv (↑A.card) ε)) 1
    ⊢ Exists fun n => And (LT.lt 0 n) (LT.lt (HDiv.hDiv ↑(Finset.filter (fun a =>  …
  -/
  have hn : 0 < n := Nat.succ_pos _
  /-
    A : Finset Nat
    ε : Real
    hε : LT.lt 0 ε
    hε₁ : LE.le ε 1
    n : Nat := HAdd.hAdd (Nat.floor (HDiv.hDiv (↑A.card) ε)) 1
    hn : LT.lt 0 n
    ⊢ Exists fun n => And (LT.lt 0 n) (LT.lt (HDiv.hDiv ↑(Finset.filter (fun a =>  …
  -/
  use n, hn
  /-
    case right
    A : Finset Nat
    ε : Real
    hε : LT.lt 0 ε
    hε₁ : LE.le ε 1
    n : Nat := HAdd.hAdd (Nat.floor (HDiv.hDiv (↑A.card) ε)) 1
    hn : LT.lt 0 n
    ⊢ LT.lt (HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem (↑A) a) (Finset.Io …
  -/
  rw [div_lt_iff₀ (Nat.cast_pos.2 hn), ← div_lt_iff₀' hε, Nat.cast_add_one]
  /-
    case right
    A : Finset Nat
    ε : Real
    hε : LT.lt 0 ε
    hε₁ : LE.le ε 1
    n : Nat := HAdd.hAdd (Nat.floor (HDiv.hDiv (↑A.card) ε)) 1
    hn : LT.lt 0 n
    ⊢ LT.lt (HDiv.hDiv (↑(Finset.filter (fun a => Membership.mem (↑A) a) (Finset.I …
  -/
  exact (Nat.lt_floor_add_one _).trans_le' <| by gcongr; simp [subset_iff]
  /-
    🎉 no goals
  -/


/-- The Schnirelmann density of any finite set is `0`. -/
lemma schnirelmannDensity_finite {A : Set ℕ} [DecidablePred (· ∈ A)] (hA : A.Finite) :
                                    /-
                                      A : Set Nat
                                      inst✝ : DecidablePred fun x => Membership.mem A x
                                      hA : A.Finite
                                      ⊢ Eq (schnirelmannDensity A) 0
                                    -/
    schnirelmannDensity A = 0 := by simpa using schnirelmannDensity_finset hA.toFinset
                                    /-
                                      🎉 no goals
                                    -/


@[simp] lemma schnirelmannDensity_univ : schnirelmannDensity Set.univ = 1 :=
                                                  /-
                                                    ⊢ Membership.mem Set.univ 0
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  (schnirelmannDensity_eq_one_iff_of_zero_mem (by simp)).2 (by simp)
                                                               /-
                                                                 🎉 no goals
                                                               -/


lemma schnirelmannDensity_setOf_even : schnirelmannDensity (setOf Even) = 0 :=
                                                   /-
                                                     ⊢ Not (Membership.mem (setOf Even) 1)
                                                   -/
  schnirelmannDensity_eq_zero_of_one_not_mem <| by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


lemma schnirelmannDensity_setOf_prime : schnirelmannDensity (setOf Nat.Prime) = 0 :=
                                                   /-
                                                     ⊢ Not (Membership.mem (setOf Nat.Prime) 1)
                                                   -/
  schnirelmannDensity_eq_zero_of_one_not_mem <| by simp [Nat.not_prime_one]
                                                   /-
                                                     🎉 no goals
                                                   -/


/--
The Schnirelmann density of the set of naturals which are `1 mod m` is `m⁻¹`, for any `m ≠ 1`.

Note that if `m = 1`, this set is empty.
-/
lemma schnirelmannDensity_setOf_mod_eq_one {m : ℕ} (hm : m ≠ 1) :
    schnirelmannDensity {n | n % m = 1} = (m⁻¹ : ℝ) := by
  /-
    m : Nat
    hm : Ne m 1
    ⊢ Eq (schnirelmannDensity (setOf fun n => Eq (HMod.hMod n m) 1)) (Inv.inv ↑m)
  -/
  rcases m.eq_zero_or_pos with rfl | hm'
    /-
      case inl
      hm : Ne 0 1
      ⊢ Eq (schnirelmannDensity (setOf fun n => Eq (HMod.hMod n 0) 1)) (Inv.inv ↑0)
    -/
  · simp only [Nat.cast_zero, inv_zero]
    /-
      case inl
      hm : Ne 0 1
      ⊢ Eq (schnirelmannDensity (setOf fun n => Eq (HMod.hMod n 0) 1)) 0
    -/
    refine schnirelmannDensity_finite ?_
    /-
      case inl
      hm : Ne 0 1
      ⊢ (setOf fun n => Eq (HMod.hMod n 0) 1).Finite
    -/
    simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    m : Nat
    hm : Ne m 1
    hm' : GT.gt m 0
    ⊢ Eq (schnirelmannDensity (setOf fun n => Eq (HMod.hMod n m) 1)) (Inv.inv ↑m)
  -/
  apply le_antisymm (schnirelmannDensity_le_of_le m hm'.ne' _) _
    /-
      m : Nat
      hm : Ne m 1
      hm' : GT.gt m 0
      ⊢ LE.le (HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem (setOf fun n => Eq …
    -/
  · rw [← one_div, ← @Nat.cast_one ℝ]
    /-
      m : Nat
      hm : Ne m 1
      hm' : GT.gt m 0
      ⊢ LE.le (HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem (setOf fun n => Eq …
    -/
    gcongr
    simp only [Set.mem_setOf_eq, card_le_one_iff_subset_singleton, subset_iff,
      mem_filter, mem_Ioc, mem_singleton, and_imp]
    /-
      case hab.h
      m : Nat
      hm : Ne m 1
      hm' : GT.gt m 0
      ⊢ Exists fun x => ∀ ⦃x_1 : Nat⦄, LT.lt 0 x_1 → LE.le x_1 m → Eq (HMod.hMod x_1 …
    -/
    use 1
    /-
      case h
      m : Nat
      hm : Ne m 1
      hm' : GT.gt m 0
      ⊢ ∀ ⦃x : Nat⦄, LT.lt 0 x → LE.le x m → Eq (HMod.hMod x m) 1 → Eq x 1
    -/
    intro x _ hxm h
    /-
      case h
      m : Nat
      hm : Ne m 1
      hm' : GT.gt m 0
      x : Nat
      a✝ : LT.lt 0 x
      hxm : LE.le x m
      h : Eq (HMod.hMod x m) 1
      ⊢ Eq x 1
    -/
    rcases eq_or_lt_of_le hxm with rfl | hxm'
      /-
        case h.inl
        x : Nat
        a✝ : LT.lt 0 x
        hm : Ne x 1
        hm' : GT.gt x 0
        hxm : LE.le x x
        h : Eq (HMod.hMod x x) 1
        ⊢ Eq x 1
      -/
    · simp at h
      /-
        🎉 no goals
      -/
    /-
      case h.inr
      m : Nat
      hm : Ne m 1
      hm' : GT.gt m 0
      x : Nat
      a✝ : LT.lt 0 x
      hxm : LE.le x m
      h : Eq (HMod.hMod x m) 1
      hxm' : LT.lt x m
      ⊢ Eq x 1
    -/
    rwa [Nat.mod_eq_of_lt hxm'] at h
    /-
      🎉 no goals
    -/
  /-
    m : Nat
    hm : Ne m 1
    hm' : GT.gt m 0
    ⊢ LE.le (Inv.inv ↑m) (schnirelmannDensity (setOf fun n => Eq (HMod.hMod n m) 1))
  -/
  rw [le_schnirelmannDensity_iff]
  /-
    m : Nat
    hm : Ne m 1
    hm' : GT.gt m 0
    ⊢ ∀ (n : Nat), LT.lt 0 n → LE.le (Inv.inv ↑m) (HDiv.hDiv ↑(Finset.filter (fun  …
  -/
  intro n hn
  /-
    m : Nat
    hm : Ne m 1
    hm' : GT.gt m 0
    n : Nat
    hn : LT.lt 0 n
    ⊢ LE.le (Inv.inv ↑m) (HDiv.hDiv ↑(Finset.filter (fun a => Membership.mem (setO …
  -/
  simp only [Set.mem_setOf_eq]
  have : (Icc 0 ((n - 1) / m)).image (· * m + 1) ⊆ {x ∈ Ioc 0 n | x % m = 1} := by
    simp only [subset_iff, mem_image, forall_exists_index, mem_filter, mem_Ioc, mem_Icc, and_imp]
    rintro _ y _ hy' rfl
    have hm : 2 ≤ m := hm.lt_of_le' hm'
    simp only [Nat.mul_add_mod', Nat.mod_eq_of_lt hm, add_pos_iff, or_true, and_true, true_and,
      ← Nat.le_sub_iff_add_le hn, zero_lt_one]
    exact Nat.mul_le_of_le_div _ _ _ hy'
  /-
    m : Nat
    hm : Ne m 1
    hm' : GT.gt m 0
    n : Nat
    hn : LT.lt 0 n
    this : HasSubset.Subset (Finset.image (fun x => HAdd.hAdd (HMul.hMul x m) 1) ( …
    ⊢ LE.le (Inv.inv ↑m) (HDiv.hDiv ↑(Finset.filter (fun a => Eq (HMod.hMod a m) 1 …
  -/
  rw [le_div_iff₀ (Nat.cast_pos.2 hn), mul_comm, ← div_eq_mul_inv]
  /-
    m : Nat
    hm : Ne m 1
    hm' : GT.gt m 0
    n : Nat
    hn : LT.lt 0 n
    this : HasSubset.Subset (Finset.image (fun x => HAdd.hAdd (HMul.hMul x m) 1) ( …
    ⊢ LE.le (HDiv.hDiv ↑n ↑m) ↑(Finset.filter (fun a => Eq (HMod.hMod a m) 1) (Fin …
  -/
  apply (Nat.cast_le.2 (card_le_card this)).trans'
  rw [card_image_of_injective, Nat.card_Icc, Nat.sub_zero, div_le_iff₀ (Nat.cast_pos.2 hm'),
    ← Nat.cast_mul, Nat.cast_le, add_one_mul (α := ℕ)]
    /-
      m : Nat
      hm : Ne m 1
      hm' : GT.gt m 0
      n : Nat
      hn : LT.lt 0 n
      this : HasSubset.Subset (Finset.image (fun x => HAdd.hAdd (HMul.hMul x m) 1) ( …
      ⊢ LE.le n (HAdd.hAdd (HMul.hMul (HDiv.hDiv (HSub.hSub n 1) m) m) m)
    -/
  · have := @Nat.lt_div_mul_add n.pred m hm'
    /-
      m : Nat
      hm : Ne m 1
      hm' : GT.gt m 0
      n : Nat
      hn : LT.lt 0 n
      this✝ : HasSubset.Subset (Finset.image (fun x => HAdd.hAdd (HMul.hMul x m) 1)  …
      this : LT.lt n.pred (HAdd.hAdd (HMul.hMul (HDiv.hDiv n.pred m) m) m)
      ⊢ LE.le n (HAdd.hAdd (HMul.hMul (HDiv.hDiv (HSub.hSub n 1) m) m) m)
    -/
    rwa [← Nat.succ_le, Nat.succ_pred hn.ne'] at this
    /-
      🎉 no goals
    -/
  /-
    case H
    m : Nat
    hm : Ne m 1
    hm' : GT.gt m 0
    n : Nat
    hn : LT.lt 0 n
    this : HasSubset.Subset (Finset.image (fun x => HAdd.hAdd (HMul.hMul x m) 1) ( …
    ⊢ Function.Injective fun x => HAdd.hAdd (HMul.hMul x m) 1
  -/
  intro a b
  /-
    case H
    m : Nat
    hm : Ne m 1
    hm' : GT.gt m 0
    n : Nat
    hn : LT.lt 0 n
    this : HasSubset.Subset (Finset.image (fun x => HAdd.hAdd (HMul.hMul x m) 1) ( …
    a b : Nat
    ⊢ Eq ((fun x => HAdd.hAdd (HMul.hMul x m) 1) a) ((fun x => HAdd.hAdd (HMul.hMu …
  -/
  simp [hm'.ne']
  /-
    🎉 no goals
  -/


lemma schnirelmannDensity_setOf_modeq_one {m : ℕ} :
    schnirelmannDensity {n | n ≡ 1 [MOD m]} = (m⁻¹ : ℝ) := by
  /-
    m : Nat
    ⊢ Eq (schnirelmannDensity (setOf fun n => m.ModEq n 1)) (Inv.inv ↑m)
  -/
  rcases eq_or_ne m 1 with rfl | hm
    /-
      case inl
      ⊢ Eq (schnirelmannDensity (setOf fun n => Nat.ModEq 1 n 1)) (Inv.inv ↑1)
    -/
  · simp [Nat.modEq_one]
    /-
      🎉 no goals
    -/
  /-
    case inr
    m : Nat
    hm : Ne m 1
    ⊢ Eq (schnirelmannDensity (setOf fun n => m.ModEq n 1)) (Inv.inv ↑m)
  -/
  rw [← schnirelmannDensity_setOf_mod_eq_one hm]
  /-
    case inr
    m : Nat
    hm : Ne m 1
    ⊢ Eq (schnirelmannDensity (setOf fun n => m.ModEq n 1)) (schnirelmannDensity ( …
  -/
  apply schnirelmannDensity_congr
  /-
    case inr.h
    m : Nat
    hm : Ne m 1
    ⊢ Eq (setOf fun n => m.ModEq n 1) (setOf fun n => Eq (HMod.hMod n m) 1)
  -/
  ext n
  /-
    case inr.h.h
    m : Nat
    hm : Ne m 1
    n : Nat
    ⊢ Iff (Membership.mem (setOf fun n => m.ModEq n 1) n) (Membership.mem (setOf f …
  -/
  simp only [Set.mem_setOf_eq, Nat.ModEq, Nat.one_mod_eq_one.mpr hm]
  /-
    🎉 no goals
  -/


lemma schnirelmannDensity_setOf_Odd : schnirelmannDensity (setOf Odd) = 2⁻¹ := by
  /-
    ⊢ Eq (schnirelmannDensity (setOf Odd)) (Inv.inv 2)
  -/
  have h : setOf Odd = {n | n % 2 = 1} := Set.ext fun _ => Nat.odd_iff
  /-
    h : Eq (setOf Odd) (setOf fun n => Eq (HMod.hMod n 2) 1)
    ⊢ Eq (schnirelmannDensity (setOf Odd)) (Inv.inv 2)
  -/
  simp only [h]
  /-
    h : Eq (setOf Odd) (setOf fun n => Eq (HMod.hMod n 2) 1)
    ⊢ Eq (schnirelmannDensity (setOf fun n => Eq (HMod.hMod n 2) 1)) (Inv.inv 2)
  -/
  rw [schnirelmannDensity_setOf_mod_eq_one (by norm_num1), Nat.cast_two]
  /-
    🎉 no goals
  -/

