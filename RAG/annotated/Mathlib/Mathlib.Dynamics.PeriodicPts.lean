/-- A point `x` is a periodic point of `f : α → α` of period `n` if `f^[n] x = x`.
Note that we do not require `0 < n` in this definition. Many theorems about periodic points
need this assumption. -/
def IsPeriodicPt (f : α → α) (n : ℕ) (x : α) :=
  IsFixedPt f^[n] x


/-- A fixed point of `f` is a periodic point of `f` of any prescribed period. -/
theorem IsFixedPt.isPeriodicPt (hf : IsFixedPt f x) (n : ℕ) : IsPeriodicPt f n x :=
  hf.iterate n


/-- For the identity map, all points are periodic. -/
theorem is_periodic_id (n : ℕ) (x : α) : IsPeriodicPt id n x :=
  (isFixedPt_id x).isPeriodicPt n


/-- Any point is a periodic point of period `0`. -/
theorem isPeriodicPt_zero (f : α → α) (x : α) : IsPeriodicPt f 0 x :=
  isFixedPt_id x


instance [DecidableEq α] {f : α → α} {n : ℕ} {x : α} : Decidable (IsPeriodicPt f n x) :=
  IsFixedPt.decidable


protected theorem isFixedPt (hf : IsPeriodicPt f n x) : IsFixedPt f^[n] x :=
  hf


protected theorem map (hx : IsPeriodicPt fa n x) {g : α → β} (hg : Semiconj g fa fb) :
    IsPeriodicPt fb n (g x) :=
  IsFixedPt.map hx (hg.iterate_right n)


theorem apply_iterate (hx : IsPeriodicPt f n x) (m : ℕ) : IsPeriodicPt f n (f^[m] x) :=
  hx.map <| Commute.iterate_self f m


protected theorem apply (hx : IsPeriodicPt f n x) : IsPeriodicPt f n (f x) :=
  hx.apply_iterate 1


protected theorem add (hn : IsPeriodicPt f n x) (hm : IsPeriodicPt f m x) :
    IsPeriodicPt f (n + m) x := by
  /-
    α : Type u_1
    f : α → α
    x : α
    m n : Nat
    hn : Function.IsPeriodicPt f n x
    hm : Function.IsPeriodicPt f m x
    ⊢ Function.IsPeriodicPt f (HAdd.hAdd n m) x
  -/
  rw [IsPeriodicPt, iterate_add]
  /-
    α : Type u_1
    f : α → α
    x : α
    m n : Nat
    hn : Function.IsPeriodicPt f n x
    hm : Function.IsPeriodicPt f m x
    ⊢ Function.IsFixedPt (Function.comp (Nat.iterate f n) (Nat.iterate f m)) x
  -/
  exact hn.comp hm
  /-
    🎉 no goals
  -/


theorem left_of_add (hn : IsPeriodicPt f (n + m) x) (hm : IsPeriodicPt f m x) :
    IsPeriodicPt f n x := by
  /-
    α : Type u_1
    f : α → α
    x : α
    m n : Nat
    hn : Function.IsPeriodicPt f (HAdd.hAdd n m) x
    hm : Function.IsPeriodicPt f m x
    ⊢ Function.IsPeriodicPt f n x
  -/
  rw [IsPeriodicPt, iterate_add] at hn
  /-
    α : Type u_1
    f : α → α
    x : α
    m n : Nat
    hn : Function.IsFixedPt (Function.comp (Nat.iterate f n) (Nat.iterate f m)) x
    hm : Function.IsPeriodicPt f m x
    ⊢ Function.IsPeriodicPt f n x
  -/
  exact hn.left_of_comp hm
  /-
    🎉 no goals
  -/


theorem right_of_add (hn : IsPeriodicPt f (n + m) x) (hm : IsPeriodicPt f n x) :
    IsPeriodicPt f m x := by
  /-
    α : Type u_1
    f : α → α
    x : α
    m n : Nat
    hn : Function.IsPeriodicPt f (HAdd.hAdd n m) x
    hm : Function.IsPeriodicPt f n x
    ⊢ Function.IsPeriodicPt f m x
  -/
  rw [add_comm] at hn
  /-
    α : Type u_1
    f : α → α
    x : α
    m n : Nat
    hn : Function.IsPeriodicPt f (HAdd.hAdd m n) x
    hm : Function.IsPeriodicPt f n x
    ⊢ Function.IsPeriodicPt f m x
  -/
  exact hn.left_of_add hm
  /-
    🎉 no goals
  -/


protected theorem sub (hm : IsPeriodicPt f m x) (hn : IsPeriodicPt f n x) :
    IsPeriodicPt f (m - n) x := by
  /-
    α : Type u_1
    f : α → α
    x : α
    m n : Nat
    hm : Function.IsPeriodicPt f m x
    hn : Function.IsPeriodicPt f n x
    ⊢ Function.IsPeriodicPt f (HSub.hSub m n) x
  -/
  rcases le_total n m with h | h
    /-
      case inl
      α : Type u_1
      f : α → α
      x : α
      m n : Nat
      hm : Function.IsPeriodicPt f m x
      hn : Function.IsPeriodicPt f n x
      h : LE.le n m
      ⊢ Function.IsPeriodicPt f (HSub.hSub m n) x
    -/
  · refine left_of_add ?_ hn
    /-
      case inl
      α : Type u_1
      f : α → α
      x : α
      m n : Nat
      hm : Function.IsPeriodicPt f m x
      hn : Function.IsPeriodicPt f n x
      h : LE.le n m
      ⊢ Function.IsPeriodicPt f (HAdd.hAdd (HSub.hSub m n) n) x
    -/
    rwa [tsub_add_cancel_of_le h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      f : α → α
      x : α
      m n : Nat
      hm : Function.IsPeriodicPt f m x
      hn : Function.IsPeriodicPt f n x
      h : LE.le m n
      ⊢ Function.IsPeriodicPt f (HSub.hSub m n) x
    -/
  · rw [tsub_eq_zero_iff_le.mpr h]
    /-
      case inr
      α : Type u_1
      f : α → α
      x : α
      m n : Nat
      hm : Function.IsPeriodicPt f m x
      hn : Function.IsPeriodicPt f n x
      h : LE.le m n
      ⊢ Function.IsPeriodicPt f 0 x
    -/
    apply isPeriodicPt_zero
    /-
      🎉 no goals
    -/


protected theorem mul_const (hm : IsPeriodicPt f m x) (n : ℕ) : IsPeriodicPt f (m * n) x := by
  /-
    α : Type u_1
    f : α → α
    x : α
    m : Nat
    hm : Function.IsPeriodicPt f m x
    n : Nat
    ⊢ Function.IsPeriodicPt f (HMul.hMul m n) x
  -/
  simp only [IsPeriodicPt, iterate_mul, hm.isFixedPt.iterate n]
  /-
    🎉 no goals
  -/


protected theorem const_mul (hm : IsPeriodicPt f m x) (n : ℕ) : IsPeriodicPt f (n * m) x := by
  /-
    α : Type u_1
    f : α → α
    x : α
    m : Nat
    hm : Function.IsPeriodicPt f m x
    n : Nat
    ⊢ Function.IsPeriodicPt f (HMul.hMul n m) x
  -/
  simp only [mul_comm n, hm.mul_const n]
  /-
    🎉 no goals
  -/


theorem trans_dvd (hm : IsPeriodicPt f m x) {n : ℕ} (hn : m ∣ n) : IsPeriodicPt f n x :=
  let ⟨k, hk⟩ := hn
  hk.symm ▸ hm.mul_const k


protected theorem iterate (hf : IsPeriodicPt f n x) (m : ℕ) : IsPeriodicPt f^[m] n x := by
  /-
    α : Type u_1
    f : α → α
    x : α
    n : Nat
    hf : Function.IsPeriodicPt f n x
    m : Nat
    ⊢ Function.IsPeriodicPt (Nat.iterate f m) n x
  -/
  rw [IsPeriodicPt, ← iterate_mul, mul_comm, iterate_mul]
  /-
    α : Type u_1
    f : α → α
    x : α
    n : Nat
    hf : Function.IsPeriodicPt f n x
    m : Nat
    ⊢ Function.IsFixedPt (Nat.iterate (Nat.iterate f n) m) x
  -/
  exact hf.isFixedPt.iterate m
  /-
    🎉 no goals
  -/


theorem comp {g : α → α} (hco : Commute f g) (hf : IsPeriodicPt f n x) (hg : IsPeriodicPt g n x) :
    IsPeriodicPt (f ∘ g) n x := by
  /-
    α : Type u_1
    f : α → α
    x : α
    n : Nat
    g : α → α
    hco : Function.Commute f g
    hf : Function.IsPeriodicPt f n x
    hg : Function.IsPeriodicPt g n x
    ⊢ Function.IsPeriodicPt (Function.comp f g) n x
  -/
  rw [IsPeriodicPt, hco.comp_iterate]
  /-
    α : Type u_1
    f : α → α
    x : α
    n : Nat
    g : α → α
    hco : Function.Commute f g
    hf : Function.IsPeriodicPt f n x
    hg : Function.IsPeriodicPt g n x
    ⊢ Function.IsFixedPt (Function.comp (Nat.iterate f n) (Nat.iterate g n)) x
  -/
  exact IsFixedPt.comp hf hg
  /-
    🎉 no goals
  -/


theorem comp_lcm {g : α → α} (hco : Commute f g) (hf : IsPeriodicPt f m x)
    (hg : IsPeriodicPt g n x) : IsPeriodicPt (f ∘ g) (Nat.lcm m n) x :=
  (hf.trans_dvd <| Nat.dvd_lcm_left _ _).comp hco (hg.trans_dvd <| Nat.dvd_lcm_right _ _)


theorem left_of_comp {g : α → α} (hco : Commute f g) (hfg : IsPeriodicPt (f ∘ g) n x)
    (hg : IsPeriodicPt g n x) : IsPeriodicPt f n x := by
  /-
    α : Type u_1
    f : α → α
    x : α
    n : Nat
    g : α → α
    hco : Function.Commute f g
    hfg : Function.IsPeriodicPt (Function.comp f g) n x
    hg : Function.IsPeriodicPt g n x
    ⊢ Function.IsPeriodicPt f n x
  -/
  rw [IsPeriodicPt, hco.comp_iterate] at hfg
  /-
    α : Type u_1
    f : α → α
    x : α
    n : Nat
    g : α → α
    hco : Function.Commute f g
    hfg : Function.IsFixedPt (Function.comp (Nat.iterate f n) (Nat.iterate g n)) x
    hg : Function.IsPeriodicPt g n x
    ⊢ Function.IsPeriodicPt f n x
  -/
  exact hfg.left_of_comp hg
  /-
    🎉 no goals
  -/


theorem iterate_mod_apply (h : IsPeriodicPt f n x) (m : ℕ) : f^[m % n] x = f^[m] x := by
  /-
    α : Type u_1
    f : α → α
    x : α
    n : Nat
    h : Function.IsPeriodicPt f n x
    m : Nat
    ⊢ Eq (Nat.iterate f (HMod.hMod m n) x) (Nat.iterate f m x)
  -/
  conv_rhs => rw [← Nat.mod_add_div m n, iterate_add_apply, (h.mul_const _).eq]
  /-
    🎉 no goals
  -/


protected theorem mod (hm : IsPeriodicPt f m x) (hn : IsPeriodicPt f n x) :
    IsPeriodicPt f (m % n) x :=
  (hn.iterate_mod_apply m).trans hm


protected theorem gcd (hm : IsPeriodicPt f m x) (hn : IsPeriodicPt f n x) :
    IsPeriodicPt f (m.gcd n) x := by
  /-
    α : Type u_1
    f : α → α
    x : α
    m n : Nat
    hm : Function.IsPeriodicPt f m x
    hn : Function.IsPeriodicPt f n x
    ⊢ Function.IsPeriodicPt f (m.gcd n) x
  -/
  revert hm hn
  /-
    α : Type u_1
    f : α → α
    x : α
    m n : Nat
    ⊢ Function.IsPeriodicPt f m x → Function.IsPeriodicPt f n x → Function.IsPerio …
  -/
  refine Nat.gcd.induction m n (fun n _ hn => ?_) fun m n _ ih hm hn => ?_
    /-
      case refine_1
      α : Type u_1
      f : α → α
      x : α
      m n✝ n : Nat
      x✝ : Function.IsPeriodicPt f 0 x
      hn : Function.IsPeriodicPt f n x
      ⊢ Function.IsPeriodicPt f (Nat.gcd 0 n) x
    -/
  · rwa [Nat.gcd_zero_left]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      f : α → α
      x : α
      m✝ n✝ m n : Nat
      x✝ : LT.lt 0 m
      ih : Function.IsPeriodicPt f (HMod.hMod n m) x → Function.IsPeriodicPt f m x → …
      hm : Function.IsPeriodicPt f m x
      hn : Function.IsPeriodicPt f n x
      ⊢ Function.IsPeriodicPt f (m.gcd n) x
    -/
  · rw [Nat.gcd_rec]
    /-
      case refine_2
      α : Type u_1
      f : α → α
      x : α
      m✝ n✝ m n : Nat
      x✝ : LT.lt 0 m
      ih : Function.IsPeriodicPt f (HMod.hMod n m) x → Function.IsPeriodicPt f m x → …
      hm : Function.IsPeriodicPt f m x
      hn : Function.IsPeriodicPt f n x
      ⊢ Function.IsPeriodicPt f ((HMod.hMod n m).gcd m) x
    -/
    exact ih (hn.mod hm) hm
    /-
      🎉 no goals
    -/


/-- If `f` sends two periodic points `x` and `y` of the same positive period to the same point,
then `x = y`. For a similar statement about points of different periods see `eq_of_apply_eq`. -/
theorem eq_of_apply_eq_same (hx : IsPeriodicPt f n x) (hy : IsPeriodicPt f n y) (hn : 0 < n)
    (h : f x = f y) : x = y := by
  /-
    α : Type u_1
    f : α → α
    x y : α
    n : Nat
    hx : Function.IsPeriodicPt f n x
    hy : Function.IsPeriodicPt f n y
    hn : LT.lt 0 n
    h : Eq (f x) (f y)
    ⊢ Eq x y
  -/
  rw [← hx.eq, ← hy.eq, ← iterate_pred_comp_of_pos f hn, comp_apply, comp_apply, h]
  /-
    🎉 no goals
  -/


/-- If `f` sends two periodic points `x` and `y` of positive periods to the same point,
then `x = y`. -/
theorem eq_of_apply_eq (hx : IsPeriodicPt f m x) (hy : IsPeriodicPt f n y) (hm : 0 < m) (hn : 0 < n)
    (h : f x = f y) : x = y :=
  (hx.mul_const n).eq_of_apply_eq_same (hy.const_mul m) (mul_pos hm hn) h


/-- The set of periodic points of a given (possibly non-minimal) period. -/
def ptsOfPeriod (f : α → α) (n : ℕ) : Set α :=
  { x : α | IsPeriodicPt f n x }


@[simp]
theorem mem_ptsOfPeriod : x ∈ ptsOfPeriod f n ↔ IsPeriodicPt f n x :=
  Iff.rfl


theorem Semiconj.mapsTo_ptsOfPeriod {g : α → β} (h : Semiconj g fa fb) (n : ℕ) :
    MapsTo g (ptsOfPeriod fa n) (ptsOfPeriod fb n) :=
  (h.iterate_right n).mapsTo_fixedPoints


theorem bijOn_ptsOfPeriod (f : α → α) {n : ℕ} (hn : 0 < n) :
    BijOn f (ptsOfPeriod f n) (ptsOfPeriod f n) :=
  ⟨(Commute.refl f).mapsTo_ptsOfPeriod n, fun _ hx _ hy hxy => hx.eq_of_apply_eq_same hy hn hxy,
    fun x hx =>
    ⟨f^[n.pred] x, hx.apply_iterate _, by
      /-
        α : Type u_1
        f : α → α
        n : Nat
        hn : LT.lt 0 n
        x : α
        hx : Membership.mem (Function.ptsOfPeriod f n) x
        ⊢ Eq (f (Nat.iterate f n.pred x)) x
      -/
      rw [← comp_apply (f := f), comp_iterate_pred_of_pos f hn, hx.eq]⟩⟩
      /-
        🎉 no goals
      -/


theorem directed_ptsOfPeriod_pNat (f : α → α) : Directed (· ⊆ ·) fun n : ℕ+ => ptsOfPeriod f n :=
  fun m n => ⟨m * n, fun _ hx => hx.mul_const n, fun _ hx => hx.const_mul m⟩


/-- The set of periodic points of a map `f : α → α`. -/
def periodicPts (f : α → α) : Set α :=
  { x : α | ∃ n > 0, IsPeriodicPt f n x }


theorem mk_mem_periodicPts (hn : 0 < n) (hx : IsPeriodicPt f n x) : x ∈ periodicPts f :=
  ⟨n, hn, hx⟩


theorem mem_periodicPts : x ∈ periodicPts f ↔ ∃ n > 0, IsPeriodicPt f n x :=
  Iff.rfl


theorem isPeriodicPt_of_mem_periodicPts_of_isPeriodicPt_iterate (hx : x ∈ periodicPts f)
    (hm : IsPeriodicPt f m (f^[n] x)) : IsPeriodicPt f m x := by
  /-
    α : Type u_1
    f : α → α
    x : α
    m n : Nat
    hx : Membership.mem (Function.periodicPts f) x
    hm : Function.IsPeriodicPt f m (Nat.iterate f n x)
    ⊢ Function.IsPeriodicPt f m x
  -/
  rcases hx with ⟨r, hr, hr'⟩
  suffices n ≤ (n / r + 1) * r by
    -- Porting note: convert used to unfold IsPeriodicPt
    change _ = _
    convert (hm.apply_iterate ((n / r + 1) * r - n)).eq <;>
      rw [← iterate_add_apply, Nat.sub_add_cancel this, iterate_mul, (hr'.iterate _).eq]
  /-
    case intro.intro
    α : Type u_1
    f : α → α
    x : α
    m n : Nat
    hm : Function.IsPeriodicPt f m (Nat.iterate f n x)
    r : Nat
    hr : GT.gt r 0
    hr' : Function.IsPeriodicPt f r x
    ⊢ LE.le n (HMul.hMul (HAdd.hAdd (HDiv.hDiv n r) 1) r)
  -/
  rw [add_mul, one_mul]
  /-
    case intro.intro
    α : Type u_1
    f : α → α
    x : α
    m n : Nat
    hm : Function.IsPeriodicPt f m (Nat.iterate f n x)
    r : Nat
    hr : GT.gt r 0
    hr' : Function.IsPeriodicPt f r x
    ⊢ LE.le n (HAdd.hAdd (HMul.hMul (HDiv.hDiv n r) r) r)
  -/
  exact (Nat.lt_div_mul_add hr).le
  /-
    🎉 no goals
  -/


theorem bUnion_ptsOfPeriod : ⋃ n > 0, ptsOfPeriod f n = periodicPts f :=
                      /-
                        α : Type u_1
                        f : α → α
                        x : α
                        ⊢ Iff (Membership.mem (Set.iUnion fun n => Set.iUnion fun h => Function.ptsOfP …
                      -/
  Set.ext fun x => by simp [mem_periodicPts]
                      /-
                        🎉 no goals
                      -/


theorem iUnion_pNat_ptsOfPeriod : ⋃ n : ℕ+, ptsOfPeriod f n = periodicPts f :=
  iSup_subtype.trans <| bUnion_ptsOfPeriod f


theorem bijOn_periodicPts : BijOn f (periodicPts f) (periodicPts f) :=
  iUnion_pNat_ptsOfPeriod f ▸
    bijOn_iUnion_of_directed (directed_ptsOfPeriod_pNat f) fun i => bijOn_ptsOfPeriod f i.pos


theorem Semiconj.mapsTo_periodicPts {g : α → β} (h : Semiconj g fa fb) :
    MapsTo g (periodicPts fa) (periodicPts fb) := fun _ ⟨n, hn, hx⟩ => ⟨n, hn, hx.map h⟩


open scoped Classical in
/-- Minimal period of a point `x` under an endomorphism `f`. If `x` is not a periodic point of `f`,
then `minimalPeriod f x = 0`. -/
def minimalPeriod (f : α → α) (x : α) :=
  if h : x ∈ periodicPts f then Nat.find h else 0


theorem isPeriodicPt_minimalPeriod (f : α → α) (x : α) : IsPeriodicPt f (minimalPeriod f x) x := by
  classical
  delta minimalPeriod
  split_ifs with hx
  · exact (Nat.find_spec hx).2
  · exact isPeriodicPt_zero f x


@[simp]
theorem iterate_minimalPeriod : f^[minimalPeriod f x] x = x :=
  isPeriodicPt_minimalPeriod f x


@[simp]
theorem iterate_add_minimalPeriod_eq : f^[n + minimalPeriod f x] x = f^[n] x := by
  /-
    α : Type u_1
    f : α → α
    x : α
    n : Nat
    ⊢ Eq (Nat.iterate f (HAdd.hAdd n (Function.minimalPeriod f x)) x) (Nat.iterate …
  -/
  rw [iterate_add_apply]
  /-
    α : Type u_1
    f : α → α
    x : α
    n : Nat
    ⊢ Eq (Nat.iterate f n (Nat.iterate f (Function.minimalPeriod f x) x)) (Nat.ite …
  -/
  congr
  /-
    case e_a
    α : Type u_1
    f : α → α
    x : α
    n : Nat
    ⊢ Eq (Nat.iterate f (Function.minimalPeriod f x) x) x
  -/
  exact isPeriodicPt_minimalPeriod f x
  /-
    🎉 no goals
  -/


@[simp]
theorem iterate_mod_minimalPeriod_eq : f^[n % minimalPeriod f x] x = f^[n] x :=
  (isPeriodicPt_minimalPeriod f x).iterate_mod_apply n


theorem minimalPeriod_pos_of_mem_periodicPts (hx : x ∈ periodicPts f) : 0 < minimalPeriod f x := by
  classical
  simp only [minimalPeriod, dif_pos hx, (Nat.find_spec hx).1.lt]


theorem minimalPeriod_eq_zero_of_nmem_periodicPts (hx : x ∉ periodicPts f) :
                                /-
                                  α : Type u_1
                                  f : α → α
                                  x : α
                                  hx : Not (Membership.mem (Function.periodicPts f) x)
                                  ⊢ Eq (Function.minimalPeriod f x) 0
                                -/
    minimalPeriod f x = 0 := by simp only [minimalPeriod, dif_neg hx]
                                /-
                                  🎉 no goals
                                -/


theorem IsPeriodicPt.minimalPeriod_pos (hn : 0 < n) (hx : IsPeriodicPt f n x) :
    0 < minimalPeriod f x :=
  minimalPeriod_pos_of_mem_periodicPts <| mk_mem_periodicPts hn hx


theorem minimalPeriod_pos_iff_mem_periodicPts : 0 < minimalPeriod f x ↔ x ∈ periodicPts f :=
                             /-
                               α : Type u_1
                               f : α → α
                               x : α
                               h : Not (Membership.mem (Function.periodicPts f) x)
                               ⊢ Not (LT.lt 0 (Function.minimalPeriod f x))
                             -/
  ⟨not_imp_not.1 fun h => by simp only [minimalPeriod, dif_neg h, lt_irrefl 0, not_false_iff],
                             /-
                               🎉 no goals
                             -/
    minimalPeriod_pos_of_mem_periodicPts⟩


theorem minimalPeriod_eq_zero_iff_nmem_periodicPts : minimalPeriod f x = 0 ↔ x ∉ periodicPts f := by
  /-
    α : Type u_1
    f : α → α
    x : α
    ⊢ Iff (Eq (Function.minimalPeriod f x) 0) (Not (Membership.mem (Function.perio …
  -/
  rw [← minimalPeriod_pos_iff_mem_periodicPts, not_lt, nonpos_iff_eq_zero]
  /-
    🎉 no goals
  -/


theorem IsPeriodicPt.minimalPeriod_le (hn : 0 < n) (hx : IsPeriodicPt f n x) :
    minimalPeriod f x ≤ n := by
  classical
  rw [minimalPeriod, dif_pos (mk_mem_periodicPts hn hx)]
  exact Nat.find_min' (mk_mem_periodicPts hn hx) ⟨hn, hx⟩


theorem minimalPeriod_apply_iterate (hx : x ∈ periodicPts f) (n : ℕ) :
    minimalPeriod f (f^[n] x) = minimalPeriod f x := by
  apply
    (IsPeriodicPt.minimalPeriod_le (minimalPeriod_pos_of_mem_periodicPts hx) _).antisymm
      ((isPeriodicPt_of_mem_periodicPts_of_isPeriodicPt_iterate hx
            (isPeriodicPt_minimalPeriod f _)).minimalPeriod_le
        (minimalPeriod_pos_of_mem_periodicPts _))
    /-
      α : Type u_1
      f : α → α
      x : α
      hx : Membership.mem (Function.periodicPts f) x
      n : Nat
      ⊢ Function.IsPeriodicPt f (Function.minimalPeriod f x) (Nat.iterate f n x)
    -/
  · exact (isPeriodicPt_minimalPeriod f x).apply_iterate n
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      f : α → α
      x : α
      hx : Membership.mem (Function.periodicPts f) x
      n : Nat
      ⊢ Membership.mem (Function.periodicPts f) (Nat.iterate f n x)
    -/
  · rcases hx with ⟨m, hm, hx⟩
    /-
      case intro.intro
      α : Type u_1
      f : α → α
      x : α
      n m : Nat
      hm : GT.gt m 0
      hx : Function.IsPeriodicPt f m x
      ⊢ Membership.mem (Function.periodicPts f) (Nat.iterate f n x)
    -/
    exact ⟨m, hm, hx.apply_iterate n⟩
    /-
      🎉 no goals
    -/


theorem minimalPeriod_apply (hx : x ∈ periodicPts f) : minimalPeriod f (f x) = minimalPeriod f x :=
  minimalPeriod_apply_iterate hx 1


theorem le_of_lt_minimalPeriod_of_iterate_eq {m n : ℕ} (hm : m < minimalPeriod f x)
    (hmn : f^[m] x = f^[n] x) : m ≤ n := by
  /-
    α : Type u_1
    f : α → α
    x : α
    m n : Nat
    hm : LT.lt m (Function.minimalPeriod f x)
    hmn : Eq (Nat.iterate f m x) (Nat.iterate f n x)
    ⊢ LE.le m n
  -/
  by_contra! hmn'
  /-
    α : Type u_1
    f : α → α
    x : α
    m n : Nat
    hm : LT.lt m (Function.minimalPeriod f x)
    hmn : Eq (Nat.iterate f m x) (Nat.iterate f n x)
    hmn' : LT.lt n m
    ⊢ False
  -/
  rw [← Nat.add_sub_of_le hmn'.le, add_comm, iterate_add_apply] at hmn
  exact
    ((IsPeriodicPt.minimalPeriod_le (tsub_pos_of_lt hmn')
              (isPeriodicPt_of_mem_periodicPts_of_isPeriodicPt_iterate
                (minimalPeriod_pos_iff_mem_periodicPts.1 ((zero_le m).trans_lt hm)) hmn)).trans
          (Nat.sub_le m n)).not_lt
      hm


theorem iterate_injOn_Iio_minimalPeriod : (Iio <| minimalPeriod f x).InjOn (f^[·] x) :=
  fun _m hm _n hn hmn ↦ (le_of_lt_minimalPeriod_of_iterate_eq hm hmn).antisymm
    (le_of_lt_minimalPeriod_of_iterate_eq hn hmn.symm)


theorem iterate_eq_iterate_iff_of_lt_minimalPeriod {m n : ℕ} (hm : m < minimalPeriod f x)
    (hn : n < minimalPeriod f x) : f^[m] x = f^[n] x ↔ m = n :=
  iterate_injOn_Iio_minimalPeriod.eq_iff hm hn


@[simp] theorem minimalPeriod_id : minimalPeriod id x = 1 :=
  ((is_periodic_id _ _).minimalPeriod_le Nat.one_pos).antisymm
    (Nat.succ_le_of_lt ((is_periodic_id _ _).minimalPeriod_pos Nat.one_pos))


theorem minimalPeriod_eq_one_iff_isFixedPt : minimalPeriod f x = 1 ↔ IsFixedPt f x := by
  /-
    α : Type u_1
    f : α → α
    x : α
    ⊢ Iff (Eq (Function.minimalPeriod f x) 1) (Function.IsFixedPt f x)
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      α : Type u_1
      f : α → α
      x : α
      h : Eq (Function.minimalPeriod f x) 1
      ⊢ Function.IsFixedPt f x
    -/
  · rw [← iterate_one f]
    /-
      case refine_1
      α : Type u_1
      f : α → α
      x : α
      h : Eq (Function.minimalPeriod f x) 1
      ⊢ Function.IsFixedPt (Nat.iterate f 1) x
    -/
    refine Function.IsPeriodicPt.isFixedPt ?_
    /-
      case refine_1
      α : Type u_1
      f : α → α
      x : α
      h : Eq (Function.minimalPeriod f x) 1
      ⊢ Function.IsPeriodicPt f 1 x
    -/
    rw [← h]
    /-
      case refine_1
      α : Type u_1
      f : α → α
      x : α
      h : Eq (Function.minimalPeriod f x) 1
      ⊢ Function.IsPeriodicPt f (Function.minimalPeriod f x) x
    -/
    exact isPeriodicPt_minimalPeriod f x
    /-
      🎉 no goals
    -/
  · exact
      ((h.isPeriodicPt 1).minimalPeriod_le Nat.one_pos).antisymm
        (Nat.succ_le_of_lt ((h.isPeriodicPt 1).minimalPeriod_pos Nat.one_pos))


theorem IsPeriodicPt.eq_zero_of_lt_minimalPeriod (hx : IsPeriodicPt f n x)
    (hn : n < minimalPeriod f x) : n = 0 :=
  Eq.symm <|
    (eq_or_lt_of_le <| n.zero_le).resolve_right fun hn0 => not_lt.2 (hx.minimalPeriod_le hn0) hn


theorem not_isPeriodicPt_of_pos_of_lt_minimalPeriod :
    ∀ {n : ℕ} (_ : n ≠ 0) (_ : n < minimalPeriod f x), ¬IsPeriodicPt f n x
  | 0, n0, _ => (n0 rfl).elim
  | _ + 1, _, hn => fun hp => Nat.succ_ne_zero _ (hp.eq_zero_of_lt_minimalPeriod hn)


theorem IsPeriodicPt.minimalPeriod_dvd (hx : IsPeriodicPt f n x) : minimalPeriod f x ∣ n :=
  (eq_or_lt_of_le <| n.zero_le).elim (fun hn0 => hn0 ▸ dvd_zero _) fun hn0 =>
    -- Porting note: `Nat.dvd_iff_mod_eq_zero` gained explicit arguments
    Nat.dvd_iff_mod_eq_zero.2 <|
      (hx.mod <| isPeriodicPt_minimalPeriod f x).eq_zero_of_lt_minimalPeriod <|
        Nat.mod_lt _ <| hx.minimalPeriod_pos hn0


theorem isPeriodicPt_iff_minimalPeriod_dvd : IsPeriodicPt f n x ↔ minimalPeriod f x ∣ n :=
  ⟨IsPeriodicPt.minimalPeriod_dvd, fun h => (isPeriodicPt_minimalPeriod f x).trans_dvd h⟩


theorem minimalPeriod_eq_minimalPeriod_iff {g : β → β} {y : β} :
    minimalPeriod f x = minimalPeriod g y ↔ ∀ n, IsPeriodicPt f n x ↔ IsPeriodicPt g n y := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → α
    x : α
    g : β → β
    y : β
    ⊢ Iff (Eq (Function.minimalPeriod f x) (Function.minimalPeriod g y)) (∀ (n : N …
  -/
  simp_rw [isPeriodicPt_iff_minimalPeriod_dvd, dvd_right_iff_eq]
  /-
    🎉 no goals
  -/


theorem minimalPeriod_eq_prime {p : ℕ} [hp : Fact p.Prime] (hper : IsPeriodicPt f p x)
    (hfix : ¬IsFixedPt f x) : minimalPeriod f x = p :=
  (hp.out.eq_one_or_self_of_dvd _ hper.minimalPeriod_dvd).resolve_left
    (mt minimalPeriod_eq_one_iff_isFixedPt.1 hfix)


theorem minimalPeriod_eq_prime_pow {p k : ℕ} [hp : Fact p.Prime] (hk : ¬IsPeriodicPt f (p ^ k) x)
    (hk1 : IsPeriodicPt f (p ^ (k + 1)) x) : minimalPeriod f x = p ^ (k + 1) := by
  /-
    α : Type u_1
    f : α → α
    x : α
    p k : Nat
    hp : Fact (Nat.Prime p)
    hk : Not (Function.IsPeriodicPt f (HPow.hPow p k) x)
    hk1 : Function.IsPeriodicPt f (HPow.hPow p (HAdd.hAdd k 1)) x
    ⊢ Eq (Function.minimalPeriod f x) (HPow.hPow p (HAdd.hAdd k 1))
  -/
  apply Nat.eq_prime_pow_of_dvd_least_prime_pow hp.out <;>
    /-
      case h₁
      α : Type u_1
      f : α → α
      x : α
      p k : Nat
      hp : Fact (Nat.Prime p)
      hk : Not (Function.IsPeriodicPt f (HPow.hPow p k) x)
      hk1 : Function.IsPeriodicPt f (HPow.hPow p (HAdd.hAdd k 1)) x
      ⊢ Not (Dvd.dvd (Function.minimalPeriod f x) (HPow.hPow p k))
    -/
    /-
      🎉 no goals
    -/
    rwa [← isPeriodicPt_iff_minimalPeriod_dvd]
    /-
      🎉 no goals
    -/


theorem Commute.minimalPeriod_of_comp_dvd_lcm {g : α → α} (h : Commute f g) :
    minimalPeriod (f ∘ g) x ∣ Nat.lcm (minimalPeriod f x) (minimalPeriod g x) := by
  /-
    α : Type u_1
    f : α → α
    x : α
    g : α → α
    h : Function.Commute f g
    ⊢ Dvd.dvd (Function.minimalPeriod (Function.comp f g) x) ((Function.minimalPer …
  -/
  rw [← isPeriodicPt_iff_minimalPeriod_dvd]
  /-
    α : Type u_1
    f : α → α
    x : α
    g : α → α
    h : Function.Commute f g
    ⊢ Function.IsPeriodicPt (Function.comp f g) ((Function.minimalPeriod f x).lcm  …
  -/
  exact (isPeriodicPt_minimalPeriod f x).comp_lcm h (isPeriodicPt_minimalPeriod g x)
  /-
    🎉 no goals
  -/


theorem Commute.minimalPeriod_of_comp_dvd_mul {g : α → α} (h : Commute f g) :
    minimalPeriod (f ∘ g) x ∣ minimalPeriod f x * minimalPeriod g x :=
  dvd_trans h.minimalPeriod_of_comp_dvd_lcm (lcm_dvd_mul _ _)


theorem Commute.minimalPeriod_of_comp_eq_mul_of_coprime {g : α → α} (h : Commute f g)
    (hco : Coprime (minimalPeriod f x) (minimalPeriod g x)) :
    minimalPeriod (f ∘ g) x = minimalPeriod f x * minimalPeriod g x := by
  /-
    α : Type u_1
    f : α → α
    x : α
    g : α → α
    h : Function.Commute f g
    hco : (Function.minimalPeriod f x).Coprime (Function.minimalPeriod g x)
    ⊢ Eq (Function.minimalPeriod (Function.comp f g) x) (HMul.hMul (Function.minim …
  -/
  apply h.minimalPeriod_of_comp_dvd_mul.antisymm
  suffices
    ∀ {f g : α → α},
      Commute f g →
        Coprime (minimalPeriod f x) (minimalPeriod g x) →
          minimalPeriod f x ∣ minimalPeriod (f ∘ g) x from
    hco.mul_dvd_of_dvd_of_dvd (this h hco) (h.comp_eq.symm ▸ this h.symm hco.symm)
  /-
    α : Type u_1
    f : α → α
    x : α
    g : α → α
    h : Function.Commute f g
    hco : (Function.minimalPeriod f x).Coprime (Function.minimalPeriod g x)
    ⊢ ∀ {f g : α → α}, Function.Commute f g → (Function.minimalPeriod f x).Coprime …
  -/
  intro f g h hco
  /-
    α : Type u_1
    f✝ : α → α
    x : α
    g✝ : α → α
    h✝ : Function.Commute f✝ g✝
    hco✝ : (Function.minimalPeriod f✝ x).Coprime (Function.minimalPeriod g✝ x)
    f g : α → α
    h : Function.Commute f g
    hco : (Function.minimalPeriod f x).Coprime (Function.minimalPeriod g x)
    ⊢ Dvd.dvd (Function.minimalPeriod f x) (Function.minimalPeriod (Function.comp  …
  -/
  refine hco.dvd_of_dvd_mul_left (IsPeriodicPt.left_of_comp h ?_ ?_).minimalPeriod_dvd
    /-
      case refine_1
      α : Type u_1
      f✝ : α → α
      x : α
      g✝ : α → α
      h✝ : Function.Commute f✝ g✝
      hco✝ : (Function.minimalPeriod f✝ x).Coprime (Function.minimalPeriod g✝ x)
      f g : α → α
      h : Function.Commute f g
      hco : (Function.minimalPeriod f x).Coprime (Function.minimalPeriod g x)
      ⊢ Function.IsPeriodicPt (Function.comp f g) (HMul.hMul (Function.minimalPeriod …
    -/
  · exact (isPeriodicPt_minimalPeriod _ _).const_mul _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      f✝ : α → α
      x : α
      g✝ : α → α
      h✝ : Function.Commute f✝ g✝
      hco✝ : (Function.minimalPeriod f✝ x).Coprime (Function.minimalPeriod g✝ x)
      f g : α → α
      h : Function.Commute f g
      hco : (Function.minimalPeriod f x).Coprime (Function.minimalPeriod g x)
      ⊢ Function.IsPeriodicPt g (HMul.hMul (Function.minimalPeriod g x) (Function.mi …
    -/
  · exact (isPeriodicPt_minimalPeriod _ _).mul_const _
    /-
      🎉 no goals
    -/


private theorem minimalPeriod_iterate_eq_div_gcd_aux (h : 0 < gcd (minimalPeriod f x) n) :
    minimalPeriod f^[n] x = minimalPeriod f x / Nat.gcd (minimalPeriod f x) n := by
  /-
    α : Type u_1
    f : α → α
    x : α
    n : Nat
    h : LT.lt 0 ((Function.minimalPeriod f x).gcd n)
    ⊢ Eq (Function.minimalPeriod (Nat.iterate f n) x) (HDiv.hDiv (Function.minimal …
  -/
  apply Nat.dvd_antisymm
    /-
      case a
      α : Type u_1
      f : α → α
      x : α
      n : Nat
      h : LT.lt 0 ((Function.minimalPeriod f x).gcd n)
      ⊢ Dvd.dvd (Function.minimalPeriod (Nat.iterate f n) x) (HDiv.hDiv (Function.mi …
    -/
  · apply IsPeriodicPt.minimalPeriod_dvd
    rw [IsPeriodicPt, IsFixedPt, ← iterate_mul, ← Nat.mul_div_assoc _ (gcd_dvd_left _ _),
      mul_comm, Nat.mul_div_assoc _ (gcd_dvd_right _ _), mul_comm, iterate_mul]
    /-
      case a.hx
      α : Type u_1
      f : α → α
      x : α
      n : Nat
      h : LT.lt 0 ((Function.minimalPeriod f x).gcd n)
      ⊢ Eq (Nat.iterate (Nat.iterate f (HDiv.hDiv n ((Function.minimalPeriod f x).gc …
    -/
    exact (isPeriodicPt_minimalPeriod f x).iterate _
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      f : α → α
      x : α
      n : Nat
      h : LT.lt 0 ((Function.minimalPeriod f x).gcd n)
      ⊢ Dvd.dvd (HDiv.hDiv (Function.minimalPeriod f x) ((Function.minimalPeriod f x …
    -/
  · apply Coprime.dvd_of_dvd_mul_right (coprime_div_gcd_div_gcd h)
    /-
      case a
      α : Type u_1
      f : α → α
      x : α
      n : Nat
      h : LT.lt 0 ((Function.minimalPeriod f x).gcd n)
      ⊢ Dvd.dvd (HDiv.hDiv (Function.minimalPeriod f x) ((Function.minimalPeriod f x …
    -/
    apply Nat.dvd_of_mul_dvd_mul_right h
    rw [Nat.div_mul_cancel (gcd_dvd_left _ _), mul_assoc, Nat.div_mul_cancel (gcd_dvd_right _ _),
      mul_comm]
    /-
      case a
      α : Type u_1
      f : α → α
      x : α
      n : Nat
      h : LT.lt 0 ((Function.minimalPeriod f x).gcd n)
      ⊢ Dvd.dvd (Function.minimalPeriod f x) (HMul.hMul n (Function.minimalPeriod (N …
    -/
    apply IsPeriodicPt.minimalPeriod_dvd
    /-
      case a.hx
      α : Type u_1
      f : α → α
      x : α
      n : Nat
      h : LT.lt 0 ((Function.minimalPeriod f x).gcd n)
      ⊢ Function.IsPeriodicPt f (HMul.hMul n (Function.minimalPeriod (Nat.iterate f  …
    -/
    rw [IsPeriodicPt, IsFixedPt, iterate_mul]
    /-
      case a.hx
      α : Type u_1
      f : α → α
      x : α
      n : Nat
      h : LT.lt 0 ((Function.minimalPeriod f x).gcd n)
      ⊢ Eq (Nat.iterate (Nat.iterate f n) (Function.minimalPeriod (Nat.iterate f n)  …
    -/
    exact isPeriodicPt_minimalPeriod _ _
    /-
      🎉 no goals
    -/


theorem minimalPeriod_iterate_eq_div_gcd (h : n ≠ 0) :
    minimalPeriod f^[n] x = minimalPeriod f x / Nat.gcd (minimalPeriod f x) n :=
  minimalPeriod_iterate_eq_div_gcd_aux <| gcd_pos_of_pos_right _ (Nat.pos_of_ne_zero h)


theorem minimalPeriod_iterate_eq_div_gcd' (h : x ∈ periodicPts f) :
    minimalPeriod f^[n] x = minimalPeriod f x / Nat.gcd (minimalPeriod f x) n :=
  minimalPeriod_iterate_eq_div_gcd_aux <|
    gcd_pos_of_pos_left n (minimalPeriod_pos_iff_mem_periodicPts.mpr h)


/-- The orbit of a periodic point `x` of `f` is the cycle `[x, f x, f (f x), ...]`. Its length is
the minimal period of `x`.

If `x` is not a periodic point, then this is the empty (aka nil) cycle. -/
def periodicOrbit (f : α → α) (x : α) : Cycle α :=
  (List.range (minimalPeriod f x)).map fun n => f^[n] x


/-- The definition of a periodic orbit, in terms of `List.map`. -/
theorem periodicOrbit_def (f : α → α) (x : α) :
    periodicOrbit f x = (List.range (minimalPeriod f x)).map fun n => f^[n] x :=
  rfl


/-- The definition of a periodic orbit, in terms of `Cycle.map`. -/
theorem periodicOrbit_eq_cycle_map (f : α → α) (x : α) :
    periodicOrbit f x = (List.range (minimalPeriod f x) : Cycle ℕ).map fun n => f^[n] x :=
  rfl


@[simp]
theorem periodicOrbit_length : (periodicOrbit f x).length = minimalPeriod f x := by
  /-
    α : Type u_1
    f : α → α
    x : α
    ⊢ Eq (Function.periodicOrbit f x).length (Function.minimalPeriod f x)
  -/
  rw [periodicOrbit, Cycle.length_coe, List.length_map, List.length_range]
  /-
    🎉 no goals
  -/


@[simp]
theorem periodicOrbit_eq_nil_iff_not_periodic_pt :
    periodicOrbit f x = Cycle.nil ↔ x ∉ periodicPts f := by
  /-
    α : Type u_1
    f : α → α
    x : α
    ⊢ Iff (Eq (Function.periodicOrbit f x) Cycle.nil) (Not (Membership.mem (Functi …
  -/
  simp only [periodicOrbit.eq_1, Cycle.coe_eq_nil, List.map_eq_nil_iff, List.range_eq_nil]
  /-
    α : Type u_1
    f : α → α
    x : α
    ⊢ Iff (Eq (Function.minimalPeriod f x) 0) (Not (Membership.mem (Function.perio …
  -/
  exact minimalPeriod_eq_zero_iff_nmem_periodicPts
  /-
    🎉 no goals
  -/


theorem periodicOrbit_eq_nil_of_not_periodic_pt (h : x ∉ periodicPts f) :
    periodicOrbit f x = Cycle.nil :=
  periodicOrbit_eq_nil_iff_not_periodic_pt.2 h


@[simp]
theorem mem_periodicOrbit_iff (hx : x ∈ periodicPts f) :
    y ∈ periodicOrbit f x ↔ ∃ n, f^[n] x = y := by
  /-
    α : Type u_1
    f : α → α
    x y : α
    hx : Membership.mem (Function.periodicPts f) x
    ⊢ Iff (Membership.mem (Function.periodicOrbit f x) y) (Exists fun n => Eq (Nat …
  -/
  simp only [periodicOrbit, Cycle.mem_coe_iff, List.mem_map, List.mem_range]
  /-
    α : Type u_1
    f : α → α
    x y : α
    hx : Membership.mem (Function.periodicPts f) x
    ⊢ Iff (Exists fun a => And (LT.lt a (Function.minimalPeriod f x)) (Eq (Nat.ite …
  -/
  use fun ⟨a, _, ha'⟩ => ⟨a, ha'⟩
  /-
    case mpr
    α : Type u_1
    f : α → α
    x y : α
    hx : Membership.mem (Function.periodicPts f) x
    ⊢ (Exists fun n => Eq (Nat.iterate f n x) y) → Exists fun a => And (LT.lt a (F …
  -/
  rintro ⟨n, rfl⟩
  /-
    case mpr.intro
    α : Type u_1
    f : α → α
    x : α
    hx : Membership.mem (Function.periodicPts f) x
    n : Nat
    ⊢ Exists fun a => And (LT.lt a (Function.minimalPeriod f x)) (Eq (Nat.iterate  …
  -/
  use n % minimalPeriod f x, mod_lt _ (minimalPeriod_pos_of_mem_periodicPts hx)
  /-
    case right
    α : Type u_1
    f : α → α
    x : α
    hx : Membership.mem (Function.periodicPts f) x
    n : Nat
    ⊢ Eq (Nat.iterate f (HMod.hMod n (Function.minimalPeriod f x)) x) (Nat.iterate …
  -/
  rw [iterate_mod_minimalPeriod_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem iterate_mem_periodicOrbit (hx : x ∈ periodicPts f) (n : ℕ) :
    f^[n] x ∈ periodicOrbit f x :=
  (mem_periodicOrbit_iff hx).2 ⟨n, rfl⟩


@[simp]
theorem self_mem_periodicOrbit (hx : x ∈ periodicPts f) : x ∈ periodicOrbit f x :=
  iterate_mem_periodicOrbit hx 0


theorem nodup_periodicOrbit : (periodicOrbit f x).Nodup := by
  /-
    α : Type u_1
    f : α → α
    x : α
    ⊢ (Function.periodicOrbit f x).Nodup
  -/
  rw [periodicOrbit, Cycle.nodup_coe_iff, List.nodup_map_iff_inj_on (List.nodup_range _)]
  /-
    α : Type u_1
    f : α → α
    x : α
    ⊢ ∀ (x_1 : Nat), Membership.mem (List.range (Function.minimalPeriod f x)) x_1  …
  -/
  intro m hm n hn hmn
  /-
    α : Type u_1
    f : α → α
    x : α
    m : Nat
    hm : Membership.mem (List.range (Function.minimalPeriod f x)) m
    n : Nat
    hn : Membership.mem (List.range (Function.minimalPeriod f x)) n
    hmn : Eq (Nat.iterate f m x) (Nat.iterate f n x)
    ⊢ Eq m n
  -/
  rw [List.mem_range] at hm hn
  /-
    α : Type u_1
    f : α → α
    x : α
    m : Nat
    hm : LT.lt m (Function.minimalPeriod f x)
    n : Nat
    hn : LT.lt n (Function.minimalPeriod f x)
    hmn : Eq (Nat.iterate f m x) (Nat.iterate f n x)
    ⊢ Eq m n
  -/
  rwa [iterate_eq_iterate_iff_of_lt_minimalPeriod hm hn] at hmn
  /-
    🎉 no goals
  -/


theorem periodicOrbit_apply_iterate_eq (hx : x ∈ periodicPts f) (n : ℕ) :
    periodicOrbit f (f^[n] x) = periodicOrbit f x :=
  Eq.symm <| Cycle.coe_eq_coe.2 <| .intro n <|
                     /-
                       α : Type u_1
                       f : α → α
                       x : α
                       hx : Membership.mem (Function.periodicPts f) x
                       n : Nat
                       ⊢ Eq ((List.map (fun n => Nat.iterate f n x) (List.range (Function.minimalPeri …
                     -/
    List.ext_get (by simp [minimalPeriod_apply_iterate hx]) fun m _ _ ↦ by
                     /-
                       🎉 no goals
                     -/
      /-
        α : Type u_1
        f : α → α
        x : α
        hx : Membership.mem (Function.periodicPts f) x
        n m : Nat
        x✝¹ : LT.lt m ((List.map (fun n => Nat.iterate f n x) (List.range (Function.mi …
        x✝ : LT.lt m (List.map (fun n_1 => Nat.iterate f n_1 (Nat.iterate f n x)) (Lis …
        ⊢ Eq (((List.map (fun n => Nat.iterate f n x) (List.range (Function.minimalPer …
      -/
      simp [List.getElem_rotate, iterate_add_apply]
      /-
        🎉 no goals
      -/


theorem periodicOrbit_apply_eq (hx : x ∈ periodicPts f) :
    periodicOrbit f (f x) = periodicOrbit f x :=
  periodicOrbit_apply_iterate_eq hx 1


theorem periodicOrbit_chain (r : α → α → Prop) {f : α → α} {x : α} :
    (periodicOrbit f x).Chain r ↔ ∀ n < minimalPeriod f x, r (f^[n] x) (f^[n + 1] x) := by
  /-
    α : Type u_1
    r : α → α → Prop
    f : α → α
    x : α
    ⊢ Iff (Cycle.Chain r (Function.periodicOrbit f x)) (∀ (n : Nat), LT.lt n (Func …
  -/
  by_cases hx : x ∈ periodicPts f
    /-
      case pos
      α : Type u_1
      r : α → α → Prop
      f : α → α
      x : α
      hx : Membership.mem (Function.periodicPts f) x
      ⊢ Iff (Cycle.Chain r (Function.periodicOrbit f x)) (∀ (n : Nat), LT.lt n (Func …
    -/
  · have hx' := minimalPeriod_pos_of_mem_periodicPts hx
    /-
      case pos
      α : Type u_1
      r : α → α → Prop
      f : α → α
      x : α
      hx : Membership.mem (Function.periodicPts f) x
      hx' : LT.lt 0 (Function.minimalPeriod f x)
      ⊢ Iff (Cycle.Chain r (Function.periodicOrbit f x)) (∀ (n : Nat), LT.lt n (Func …
    -/
    have hM := Nat.sub_add_cancel (succ_le_iff.2 hx')
    /-
      case pos
      α : Type u_1
      r : α → α → Prop
      f : α → α
      x : α
      hx : Membership.mem (Function.periodicPts f) x
      hx' : LT.lt 0 (Function.minimalPeriod f x)
      hM : Eq (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) (Nat. …
      ⊢ Iff (Cycle.Chain r (Function.periodicOrbit f x)) (∀ (n : Nat), LT.lt n (Func …
    -/
    rw [periodicOrbit, ← Cycle.map_coe, Cycle.chain_map, ← hM, Cycle.chain_range_succ]
    /-
      case pos
      α : Type u_1
      r : α → α → Prop
      f : α → α
      x : α
      hx : Membership.mem (Function.periodicPts f) x
      hx' : LT.lt 0 (Function.minimalPeriod f x)
      hM : Eq (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) (Nat. …
      ⊢ Iff (And (r (Nat.iterate f (HSub.hSub (Function.minimalPeriod f x) (Nat.succ …
    -/
    refine ⟨?_, fun H => ⟨?_, fun m hm => H _ (hm.trans (Nat.lt_succ_self _))⟩⟩
      /-
        case pos.refine_1
        α : Type u_1
        r : α → α → Prop
        f : α → α
        x : α
        hx : Membership.mem (Function.periodicPts f) x
        hx' : LT.lt 0 (Function.minimalPeriod f x)
        hM : Eq (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) (Nat. …
        ⊢ And (r (Nat.iterate f (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0))  …
      -/
    · rintro ⟨hr, H⟩ n hn
      /-
        case pos.refine_1.intro
        α : Type u_1
        r : α → α → Prop
        f : α → α
        x : α
        hx : Membership.mem (Function.periodicPts f) x
        hx' : LT.lt 0 (Function.minimalPeriod f x)
        hM : Eq (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) (Nat. …
        hr : r (Nat.iterate f (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) x) …
        H : ∀ (m : Nat), LT.lt m (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) …
        n : Nat
        hn : LT.lt n (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0))  …
        ⊢ r (Nat.iterate f n x) (Nat.iterate f (HAdd.hAdd n 1) x)
      -/
      cases' eq_or_lt_of_le (Nat.lt_succ_iff.1 hn) with hM' hM'
        /-
          case pos.refine_1.intro.inl
          α : Type u_1
          r : α → α → Prop
          f : α → α
          x : α
          hx : Membership.mem (Function.periodicPts f) x
          hx' : LT.lt 0 (Function.minimalPeriod f x)
          hM : Eq (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) (Nat. …
          hr : r (Nat.iterate f (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) x) …
          H : ∀ (m : Nat), LT.lt m (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) …
          n : Nat
          hn : LT.lt n (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0))  …
          hM' : Eq n (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0))
          ⊢ r (Nat.iterate f n x) (Nat.iterate f (HAdd.hAdd n 1) x)
        -/
      · rwa [hM', hM, iterate_minimalPeriod]
        /-
          🎉 no goals
        -/
        /-
          case pos.refine_1.intro.inr
          α : Type u_1
          r : α → α → Prop
          f : α → α
          x : α
          hx : Membership.mem (Function.periodicPts f) x
          hx' : LT.lt 0 (Function.minimalPeriod f x)
          hM : Eq (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) (Nat. …
          hr : r (Nat.iterate f (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) x) …
          H : ∀ (m : Nat), LT.lt m (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) …
          n : Nat
          hn : LT.lt n (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0))  …
          hM' : LT.lt n (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0))
          ⊢ r (Nat.iterate f n x) (Nat.iterate f (HAdd.hAdd n 1) x)
        -/
      · exact H _ hM'
        /-
          🎉 no goals
        -/
      /-
        case pos.refine_2
        α : Type u_1
        r : α → α → Prop
        f : α → α
        x : α
        hx : Membership.mem (Function.periodicPts f) x
        hx' : LT.lt 0 (Function.minimalPeriod f x)
        hM : Eq (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) (Nat. …
        H : ∀ (n : Nat), LT.lt n (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (N …
        ⊢ r (Nat.iterate f (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) x) (N …
      -/
    · rw [iterate_zero_apply]
      /-
        case pos.refine_2
        α : Type u_1
        r : α → α → Prop
        f : α → α
        x : α
        hx : Membership.mem (Function.periodicPts f) x
        hx' : LT.lt 0 (Function.minimalPeriod f x)
        hM : Eq (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) (Nat. …
        H : ∀ (n : Nat), LT.lt n (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (N …
        ⊢ r (Nat.iterate f (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) x) x
      -/
      nth_rw 3 [← @iterate_minimalPeriod α f x]
      /-
        case pos.refine_2
        α : Type u_1
        r : α → α → Prop
        f : α → α
        x : α
        hx : Membership.mem (Function.periodicPts f) x
        hx' : LT.lt 0 (Function.minimalPeriod f x)
        hM : Eq (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) (Nat. …
        H : ∀ (n : Nat), LT.lt n (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (N …
        ⊢ r (Nat.iterate f (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) x) (N …
      -/
      nth_rw 2 [← hM]
      /-
        case pos.refine_2
        α : Type u_1
        r : α → α → Prop
        f : α → α
        x : α
        hx : Membership.mem (Function.periodicPts f) x
        hx' : LT.lt 0 (Function.minimalPeriod f x)
        hM : Eq (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) (Nat. …
        H : ∀ (n : Nat), LT.lt n (HAdd.hAdd (HSub.hSub (Function.minimalPeriod f x) (N …
        ⊢ r (Nat.iterate f (HSub.hSub (Function.minimalPeriod f x) (Nat.succ 0)) x) (N …
      -/
      exact H _ (Nat.lt_succ_self _)
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      r : α → α → Prop
      f : α → α
      x : α
      hx : Not (Membership.mem (Function.periodicPts f) x)
      ⊢ Iff (Cycle.Chain r (Function.periodicOrbit f x)) (∀ (n : Nat), LT.lt n (Func …
    -/
  · rw [periodicOrbit_eq_nil_of_not_periodic_pt hx, minimalPeriod_eq_zero_of_nmem_periodicPts hx]
    /-
      case neg
      α : Type u_1
      r : α → α → Prop
      f : α → α
      x : α
      hx : Not (Membership.mem (Function.periodicPts f) x)
      ⊢ Iff (Cycle.Chain r Cycle.nil) (∀ (n : Nat), LT.lt n 0 → r (Nat.iterate f n x …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem periodicOrbit_chain' (r : α → α → Prop) {f : α → α} {x : α} (hx : x ∈ periodicPts f) :
    (periodicOrbit f x).Chain r ↔ ∀ n, r (f^[n] x) (f^[n + 1] x) := by
  /-
    α : Type u_1
    r : α → α → Prop
    f : α → α
    x : α
    hx : Membership.mem (Function.periodicPts f) x
    ⊢ Iff (Cycle.Chain r (Function.periodicOrbit f x)) (∀ (n : Nat), r (Nat.iterat …
  -/
  rw [periodicOrbit_chain r]
  /-
    α : Type u_1
    r : α → α → Prop
    f : α → α
    x : α
    hx : Membership.mem (Function.periodicPts f) x
    ⊢ Iff (∀ (n : Nat), LT.lt n (Function.minimalPeriod f x) → r (Nat.iterate f n  …
  -/
  refine ⟨fun H n => ?_, fun H n _ => H n⟩
  rw [iterate_succ_apply, ← iterate_mod_minimalPeriod_eq, ← iterate_mod_minimalPeriod_eq (n := n),
    ← iterate_succ_apply, minimalPeriod_apply hx]
  /-
    α : Type u_1
    r : α → α → Prop
    f : α → α
    x : α
    hx : Membership.mem (Function.periodicPts f) x
    H : ∀ (n : Nat), LT.lt n (Function.minimalPeriod f x) → r (Nat.iterate f n x)  …
    n : Nat
    ⊢ r (Nat.iterate f (HMod.hMod n (Function.minimalPeriod f x)) x) (Nat.iterate  …
  -/
  exact H _ (mod_lt _ (minimalPeriod_pos_of_mem_periodicPts hx))
  /-
    🎉 no goals
  -/


@[simp]
theorem isFixedPt_prod_map (x : α × β) :
    IsFixedPt (Prod.map f g) x ↔ IsFixedPt f x.1 ∧ IsFixedPt g x.2 :=
  Prod.ext_iff


@[simp]
theorem isPeriodicPt_prod_map (x : α × β) :
    IsPeriodicPt (Prod.map f g) n x ↔ IsPeriodicPt f n x.1 ∧ IsPeriodicPt g n x.2 := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → α
    g : β → β
    n : Nat
    x : Prod α β
    ⊢ Iff (Function.IsPeriodicPt (Prod.map f g) n x) (And (Function.IsPeriodicPt f …
  -/
  simp [IsPeriodicPt]
  /-
    🎉 no goals
  -/


theorem minimalPeriod_prod_map (f : α → α) (g : β → β) (x : α × β) :
    minimalPeriod (Prod.map f g) x = (minimalPeriod f x.1).lcm (minimalPeriod g x.2) :=
                         /-
                           α : Type u_1
                           β : Type u_2
                           f : α → α
                           g : β → β
                           x : Prod α β
                           ⊢ ∀ (c : Nat), Iff (Dvd.dvd (Function.minimalPeriod (Prod.map f g) x) c) (Dvd. …
                         -/
  eq_of_forall_dvd <| by cases x; simp [← isPeriodicPt_iff_minimalPeriod_dvd, Nat.lcm_dvd_iff]
                                  /-
                                    🎉 no goals
                                  -/


theorem minimalPeriod_fst_dvd : minimalPeriod f x.1 ∣ minimalPeriod (Prod.map f g) x := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → α
    g : β → β
    x : Prod α β
    ⊢ Dvd.dvd (Function.minimalPeriod f x.1) (Function.minimalPeriod (Prod.map f g …
  -/
  rw [minimalPeriod_prod_map]; exact Nat.dvd_lcm_left _ _
                               /-
                                 🎉 no goals
                               -/


theorem minimalPeriod_snd_dvd : minimalPeriod g x.2 ∣ minimalPeriod (Prod.map f g) x := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → α
    g : β → β
    x : Prod α β
    ⊢ Dvd.dvd (Function.minimalPeriod g x.2) (Function.minimalPeriod (Prod.map f g …
  -/
  rw [minimalPeriod_prod_map]; exact Nat.dvd_lcm_right _ _
                               /-
                                 🎉 no goals
                               -/


/--
The period of a multiplicative action of `g` on `a` is the smallest positive `n` such that
`g ^ n • a = a`, or `0` if such an `n` does not exist.
-/
@[to_additive "The period of an additive action of `g` on `a` is the smallest positive `n`
such that `(n • g) +ᵥ a = a`, or `0` if such an `n` does not exist."]
noncomputable def period (m : M) (a : α) : ℕ := minimalPeriod (fun x => m • x) a


/-- `MulAction.period m a` is definitionally equal to `Function.minimalPeriod (m • ·) a`. -/
@[to_additive "`AddAction.period m a` is definitionally equal to
`Function.minimalPeriod (m +ᵥ ·) a`"]
theorem period_eq_minimalPeriod {m : M} {a : α} :
    MulAction.period m a = minimalPeriod (fun x => m • x) a := rfl


/-- `m ^ (period m a)` fixes `a`. -/
@[to_additive (attr := simp) "`(period m a) • m` fixes `a`."]
theorem pow_period_smul (m : M) (a : α) : m ^ (period m a) • a = a := by
  /-
    α : Type v
    M : Type u
    inst✝¹ : Monoid M
    inst✝ : MulAction M α
    m : M
    a : α
    ⊢ Eq (HSMul.hSMul (HPow.hPow m (MulAction.period m a)) a) a
  -/
  rw [period_eq_minimalPeriod, ← smul_iterate_apply, iterate_minimalPeriod]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma isPeriodicPt_smul_iff {m : M} {a : α} {n : ℕ} :
    IsPeriodicPt (m • ·) n a ↔ m ^ n • a = a := by
  /-
    α : Type v
    M : Type u
    inst✝¹ : Monoid M
    inst✝ : MulAction M α
    m : M
    a : α
    n : Nat
    ⊢ Iff (Function.IsPeriodicPt (fun x => HSMul.hSMul m x) n a) (Eq (HSMul.hSMul  …
  -/
  rw [← smul_iterate_apply, IsPeriodicPt, IsFixedPt]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem pow_smul_eq_iff_period_dvd {n : ℕ} {m : M} {a : α} :
    m ^ n • a = a ↔ period m a ∣ n := by
  /-
    α : Type v
    M : Type u
    inst✝¹ : Monoid M
    inst✝ : MulAction M α
    n : Nat
    m : M
    a : α
    ⊢ Iff (Eq (HSMul.hSMul (HPow.hPow m n) a) a) (Dvd.dvd (MulAction.period m a) n)
  -/
  rw [period_eq_minimalPeriod, ← isPeriodicPt_iff_minimalPeriod_dvd, isPeriodicPt_smul_iff]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem zpow_smul_eq_iff_period_dvd {j : ℤ} {g : G} {a : α} :
    g ^ j • a = a ↔ (period g a : ℤ) ∣ j := by
  /-
    α : Type v
    G : Type u
    inst✝¹ : Group G
    inst✝ : MulAction G α
    j : Int
    g : G
    a : α
    ⊢ Iff (Eq (HSMul.hSMul (HPow.hPow g j) a) a) (Dvd.dvd (↑(MulAction.period g a) …
  -/
  rcases j with n | n
    /-
      case ofNat
      α : Type v
      G : Type u
      inst✝¹ : Group G
      inst✝ : MulAction G α
      g : G
      a : α
      n : Nat
      ⊢ Iff (Eq (HSMul.hSMul (HPow.hPow g (Int.ofNat n)) a) a) (Dvd.dvd (↑(MulAction …
    -/
  · rw [Int.ofNat_eq_coe, zpow_natCast, Int.natCast_dvd_natCast, pow_smul_eq_iff_period_dvd]
    /-
      🎉 no goals
    -/
  · rw [Int.negSucc_coe, zpow_neg, zpow_natCast, inv_smul_eq_iff, eq_comm, dvd_neg,
      Int.natCast_dvd_natCast, pow_smul_eq_iff_period_dvd]


@[to_additive (attr := simp)]
theorem pow_mod_period_smul (n : ℕ) {m : M} {a : α} :
    m ^ (n % period m a) • a = m ^ n • a := by
  conv_rhs => rw [← Nat.mod_add_div n (period m a), pow_add, mul_smul,
    pow_smul_eq_iff_period_dvd.mpr (dvd_mul_right _ _)]


@[to_additive (attr := simp)]
theorem zpow_mod_period_smul (j : ℤ) {g : G} {a : α} :
    g ^ (j % (period g a : ℤ)) • a = g ^ j • a := by
  conv_rhs => rw [← Int.emod_add_ediv j (period g a), zpow_add, mul_smul,
    zpow_smul_eq_iff_period_dvd.mpr (dvd_mul_right _ _)]


@[to_additive (attr := simp)]
theorem pow_add_period_smul (n : ℕ) (m : M) (a : α) :
    m ^ (n + period m a) • a = m ^ n • a := by
  /-
    α : Type v
    M : Type u
    inst✝¹ : Monoid M
    inst✝ : MulAction M α
    n : Nat
    m : M
    a : α
    ⊢ Eq (HSMul.hSMul (HPow.hPow m (HAdd.hAdd n (MulAction.period m a))) a) (HSMul …
  -/
  rw [← pow_mod_period_smul, Nat.add_mod_right, pow_mod_period_smul]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem pow_period_add_smul (n : ℕ) (m : M) (a : α) :
    m ^ (period m a + n) • a = m ^ n • a := by
  /-
    α : Type v
    M : Type u
    inst✝¹ : Monoid M
    inst✝ : MulAction M α
    n : Nat
    m : M
    a : α
    ⊢ Eq (HSMul.hSMul (HPow.hPow m (HAdd.hAdd (MulAction.period m a) n)) a) (HSMul …
  -/
  rw [← pow_mod_period_smul, Nat.add_mod_left, pow_mod_period_smul]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem zpow_add_period_smul (i : ℤ) (g : G) (a : α) :
    g ^ (i + period g a) • a = g ^ i • a := by
  /-
    α : Type v
    G : Type u
    inst✝¹ : Group G
    inst✝ : MulAction G α
    i : Int
    g : G
    a : α
    ⊢ Eq (HSMul.hSMul (HPow.hPow g (HAdd.hAdd i ↑(MulAction.period g a))) a) (HSMu …
  -/
  rw [← zpow_mod_period_smul, Int.add_emod_self, zpow_mod_period_smul]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem zpow_period_add_smul (i : ℤ) (g : G) (a : α) :
    g ^ (period g a + i) • a = g ^ i • a := by
  /-
    α : Type v
    G : Type u
    inst✝¹ : Group G
    inst✝ : MulAction G α
    i : Int
    g : G
    a : α
    ⊢ Eq (HSMul.hSMul (HPow.hPow g (HAdd.hAdd (↑(MulAction.period g a)) i)) a) (HS …
  -/
  rw [← zpow_mod_period_smul, Int.add_emod_self_left, zpow_mod_period_smul]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem pow_smul_eq_iff_minimalPeriod_dvd {n : ℕ} :
    a ^ n • b = b ↔ minimalPeriod (a • ·) b ∣ n := by
  /-
    α : Type v
    G : Type u
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a : G
    b : α
    n : Nat
    ⊢ Iff (Eq (HSMul.hSMul (HPow.hPow a n) b) b) (Dvd.dvd (Function.minimalPeriod  …
  -/
  rw [← period_eq_minimalPeriod, pow_smul_eq_iff_period_dvd]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem zpow_smul_eq_iff_minimalPeriod_dvd {n : ℤ} :
    a ^ n • b = b ↔ (minimalPeriod (a • ·) b : ℤ) ∣ n := by
  /-
    α : Type v
    G : Type u
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a : G
    b : α
    n : Int
    ⊢ Iff (Eq (HSMul.hSMul (HPow.hPow a n) b) b) (Dvd.dvd (↑(Function.minimalPerio …
  -/
  rw [← period_eq_minimalPeriod, zpow_smul_eq_iff_period_dvd]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem pow_smul_mod_minimalPeriod (n : ℕ) :
    a ^ (n % minimalPeriod (a • ·) b) • b = a ^ n • b := by
  /-
    α : Type v
    G : Type u
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a : G
    b : α
    n : Nat
    ⊢ Eq (HSMul.hSMul (HPow.hPow a (HMod.hMod n (Function.minimalPeriod (fun x =>  …
  -/
  rw [← period_eq_minimalPeriod, pow_mod_period_smul]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem zpow_smul_mod_minimalPeriod (n : ℤ) :
    a ^ (n % (minimalPeriod (a • ·) b : ℤ)) • b = a ^ n • b := by
  /-
    α : Type v
    G : Type u
    inst✝¹ : Group G
    inst✝ : MulAction G α
    a : G
    b : α
    n : Int
    ⊢ Eq (HSMul.hSMul (HPow.hPow a (HMod.hMod n ↑(Function.minimalPeriod (fun x => …
  -/
  rw [← period_eq_minimalPeriod, zpow_mod_period_smul]
  /-
    🎉 no goals
  -/


