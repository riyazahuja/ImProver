local notation "π" => LieModule.toEnd R _ V


private abbrev T (w : A) : Module.End R V := (π w) - χ w • 1


/-- An auxiliary lemma used only in the definition `LieModule.weightSpaceOfIsLieTower` below. -/
private lemma weightSpaceOfIsLieTower_aux (z : L) (v : V) (hv : v ∈ weightSpace V χ) :
    ⁅z, v⁆ ∈ weightSpace V χ := by
  /-
    R : Type u_1
    L : Type u_2
    A : Type u_3
    V : Type u_4
    inst✝¹⁹ : CommRing R
    inst✝¹⁸ : IsPrincipalIdealRing R
    inst✝¹⁷ : IsDomain R
    inst✝¹⁶ : CharZero R
    inst✝¹⁵ : LieRing L
    inst✝¹⁴ : LieAlgebra R L
    inst✝¹³ : LieRing A
    inst✝¹² : LieAlgebra R A
    inst✝¹¹ : Bracket L A
    inst✝¹⁰ : Bracket A L
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : Module R V
    inst✝⁷ : Module.Free R V
    inst✝⁶ : Module.Finite R V
    inst✝⁵ : LieRingModule L V
    inst✝⁴ : LieModule R L V
    inst✝³ : LieRingModule A V
    inst✝² : LieModule R A V
    inst✝¹ : IsLieTower L A V
    inst✝ : IsLieTower A L V
    χ : A → R
    z : L
    v : V
    hv : Membership.mem (LieModule.weightSpace V χ) v
    ⊢ Membership.mem (LieModule.weightSpace V χ) (Bracket.bracket z v)
  -/
  rw [mem_weightSpace] at hv ⊢
  /-
    R : Type u_1
    L : Type u_2
    A : Type u_3
    V : Type u_4
    inst✝¹⁹ : CommRing R
    inst✝¹⁸ : IsPrincipalIdealRing R
    inst✝¹⁷ : IsDomain R
    inst✝¹⁶ : CharZero R
    inst✝¹⁵ : LieRing L
    inst✝¹⁴ : LieAlgebra R L
    inst✝¹³ : LieRing A
    inst✝¹² : LieAlgebra R A
    inst✝¹¹ : Bracket L A
    inst✝¹⁰ : Bracket A L
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : Module R V
    inst✝⁷ : Module.Free R V
    inst✝⁶ : Module.Finite R V
    inst✝⁵ : LieRingModule L V
    inst✝⁴ : LieModule R L V
    inst✝³ : LieRingModule A V
    inst✝² : LieModule R A V
    inst✝¹ : IsLieTower L A V
    inst✝ : IsLieTower A L V
    χ : A → R
    z : L
    v : V
    hv : ∀ (x : A), Eq (Bracket.bracket x v) (HSMul.hSMul (χ x) v)
    ⊢ ∀ (x : A), Eq (Bracket.bracket x (Bracket.bracket z v)) (HSMul.hSMul (χ x) ( …
  -/
  intro a
  /-
    R : Type u_1
    L : Type u_2
    A : Type u_3
    V : Type u_4
    inst✝¹⁹ : CommRing R
    inst✝¹⁸ : IsPrincipalIdealRing R
    inst✝¹⁷ : IsDomain R
    inst✝¹⁶ : CharZero R
    inst✝¹⁵ : LieRing L
    inst✝¹⁴ : LieAlgebra R L
    inst✝¹³ : LieRing A
    inst✝¹² : LieAlgebra R A
    inst✝¹¹ : Bracket L A
    inst✝¹⁰ : Bracket A L
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : Module R V
    inst✝⁷ : Module.Free R V
    inst✝⁶ : Module.Finite R V
    inst✝⁵ : LieRingModule L V
    inst✝⁴ : LieModule R L V
    inst✝³ : LieRingModule A V
    inst✝² : LieModule R A V
    inst✝¹ : IsLieTower L A V
    inst✝ : IsLieTower A L V
    χ : A → R
    z : L
    v : V
    hv : ∀ (x : A), Eq (Bracket.bracket x v) (HSMul.hSMul (χ x) v)
    a : A
    ⊢ Eq (Bracket.bracket a (Bracket.bracket z v)) (HSMul.hSMul (χ a) (Bracket.bra …
  -/
  rcases eq_or_ne v 0 with (rfl | hv')
    /-
      case inl
      R : Type u_1
      L : Type u_2
      A : Type u_3
      V : Type u_4
      inst✝¹⁹ : CommRing R
      inst✝¹⁸ : IsPrincipalIdealRing R
      inst✝¹⁷ : IsDomain R
      inst✝¹⁶ : CharZero R
      inst✝¹⁵ : LieRing L
      inst✝¹⁴ : LieAlgebra R L
      inst✝¹³ : LieRing A
      inst✝¹² : LieAlgebra R A
      inst✝¹¹ : Bracket L A
      inst✝¹⁰ : Bracket A L
      inst✝⁹ : AddCommGroup V
      inst✝⁸ : Module R V
      inst✝⁷ : Module.Free R V
      inst✝⁶ : Module.Finite R V
      inst✝⁵ : LieRingModule L V
      inst✝⁴ : LieModule R L V
      inst✝³ : LieRingModule A V
      inst✝² : LieModule R A V
      inst✝¹ : IsLieTower L A V
      inst✝ : IsLieTower A L V
      χ : A → R
      z : L
      a : A
      hv : ∀ (x : A), Eq (Bracket.bracket x 0) (HSMul.hSMul (χ x) 0)
      ⊢ Eq (Bracket.bracket a (Bracket.bracket z 0)) (HSMul.hSMul (χ a) (Bracket.bra …
    -/
  · simp only [lie_zero, smul_zero]
    /-
      🎉 no goals
    -/
  suffices χ ⁅z, a⁆ = 0 by
    rw [leibniz_lie, hv a, lie_smul, lie_swap_lie, hv, this, zero_smul, neg_zero, zero_add]
  let U' : ℕ →o Submodule R V :=
  { toFun n := Submodule.span R {((π z)^i) v | i < n},
    monotone' i j h := Submodule.span_mono (fun _ ⟨c, hc, hw⟩ ↦ ⟨c, lt_of_lt_of_le hc h, hw⟩) }
  have map_U'_le (n : ℕ) : Submodule.map (π z) (U' n) ≤ U' (n + 1) := by
    simp only [OrderHom.coe_mk, Submodule.map_span, toEnd_apply_apply, U']
    apply Submodule.span_mono
    suffices ∀ a < n, ∃ b < n + 1, ((π z) ^ b) v = ((π z) ^ (a + 1)) v by simpa [pow_succ']
    aesop
  have T_apply_succ (w : A) (n : ℕ) :
      Submodule.map (T χ w) (U' (n + 1)) ≤ U' n := by
    simp only [OrderHom.coe_mk, U', Submodule.map_span, Submodule.span_le, Set.image_subset_iff]
    simp only [Set.subset_def, Set.mem_setOf_eq, Set.mem_preimage, SetLike.mem_coe,
      forall_exists_index, and_imp, forall_apply_eq_imp_iff₂]
    induction n generalizing w
    · simp only [zero_add, Nat.lt_one_iff, LinearMap.sub_apply, LieModule.toEnd_apply_apply,
        LinearMap.smul_apply, LinearMap.one_apply, forall_eq, pow_zero, hv w, sub_self, zero_mem]
    · next n hn =>
      intro m hm
      obtain (hm | rfl) : m < n + 1 ∨ m = n + 1 := by omega
      · exact U'.mono (Nat.le_succ n) (hn w m hm)
      have H : ∀ w, ⁅w, (π z ^ n) v⁆ = (T χ w) ((π z ^ n) v) + χ w • ((π z ^ n) v) := by simp
      rw [T, LinearMap.sub_apply, pow_succ', LinearMap.mul_apply, LieModule.toEnd_apply_apply,
        LieModule.toEnd_apply_apply, LinearMap.smul_apply, LinearMap.one_apply, leibniz_lie,
        lie_swap_lie w z, H, H, lie_add, lie_smul, add_sub_assoc, add_sub_assoc, sub_self, add_zero]
      refine add_mem (neg_mem <| add_mem ?_ ?_) ?_
      · exact U'.mono n.le_succ (hn _ n n.lt_succ_self)
      · exact Submodule.smul_mem _ _ (Submodule.subset_span ⟨n, n.lt_succ_self, rfl⟩)
      · exact map_U'_le _ <| Submodule.mem_map_of_mem <| hn w n n.lt_succ_self
  set U : LieSubmodule R A V :=
  { toSubmodule := ⨆ k : ℕ, U' k
    lie_mem {w} x hx := by
      rw [show ⁅w, x⁆ = (T χ w) x + χ w • x by simp]
      apply add_mem _ (Submodule.smul_mem _ _ hx)
      set U := ⨆ k : ℕ, U' k
      suffices Submodule.map (T χ w) U ≤ U from this <| Submodule.mem_map_of_mem hx
      rw [Submodule.map_iSup, iSup_le_iff]
      rintro (_|i)
      · simp [U', Submodule.map_span]
      · exact (T_apply_succ w i).trans (le_iSup _ _) }
  have hzU (x : V) (hx : x ∈ U) : (π z) x ∈ U := by
    suffices Submodule.map (π z) U ≤ U from this <| Submodule.mem_map_of_mem hx
    simp only [U, Submodule.map_iSup, iSup_le_iff]
    exact fun i ↦ (map_U'_le i).trans (le_iSup _ _)
  have trace_za_zero : (LieModule.toEnd R A _ ⁅z, a⁆).trace R U = 0 := by
    have hres : LieModule.toEnd R A U ⁅z, a⁆ = ⁅(π z).restrict hzU, LieModule.toEnd R A U a⁆ := by
      ext ⟨x, hx⟩
      show ⁅⁅z, a⁆, x⁆ = ⁅z, ⁅a, x⁆⁆ - ⁅a, ⁅z, x⁆⁆
      simp only [leibniz_lie z a, add_sub_cancel_right]
    rw [hres, LinearMap.trace_lie]
  have trace_T_U_zero (w : A) : (T χ w).trace R U = 0 := by
    have key (i : ℕ) (hi : i ≠ 0) : ∃ j < i, Submodule.map (T χ w) (U' i) ≤ U' j := by
      obtain ⟨j, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hi
      exact ⟨j, j.lt_succ_self, T_apply_succ w j⟩
    apply IsNilpotent.eq_zero
    apply LinearMap.isNilpotent_trace_of_isNilpotent
    rw [Module.Finite.Module.End.isNilpotent_iff_of_finite]
    suffices ⨆ i, U' i ≤ Module.End.maxGenEigenspace (T χ w) 0 by
      intro x
      specialize this x.2
      simp only [Module.End.mem_maxGenEigenspace, zero_smul, sub_zero] at this
      peel this with n hn
      ext
      simp only [ZeroMemClass.coe_zero, ← hn]; clear hn
      induction n <;> simp_all [pow_succ']
    apply iSup_le
    intro i x hx
    simp only [Module.End.mem_maxGenEigenspace, zero_smul, sub_zero]
    induction i using Nat.strong_induction_on generalizing x
    next i ih =>
    obtain rfl | hi := eq_or_ne i 0
    · simp_all [U']
    obtain ⟨j, hj, hj'⟩ := key i hi
    obtain ⟨k, hk⟩ := ih j hj (hj' <| Submodule.mem_map_of_mem hx)
    use k+1
    rw [pow_succ, LinearMap.mul_apply, hk]
  have trace_za : (toEnd R A _ ⁅z, a⁆).trace R U = χ ⁅z, a⁆ • (finrank R U) := by
    simpa [T, sub_eq_zero] using trace_T_U_zero ⁅z, a⁆
  /-
    case inr
    R : Type u_1
    L : Type u_2
    A : Type u_3
    V : Type u_4
    inst✝¹⁹ : CommRing R
    inst✝¹⁸ : IsPrincipalIdealRing R
    inst✝¹⁷ : IsDomain R
    inst✝¹⁶ : CharZero R
    inst✝¹⁵ : LieRing L
    inst✝¹⁴ : LieAlgebra R L
    inst✝¹³ : LieRing A
    inst✝¹² : LieAlgebra R A
    inst✝¹¹ : Bracket L A
    inst✝¹⁰ : Bracket A L
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : Module R V
    inst✝⁷ : Module.Free R V
    inst✝⁶ : Module.Finite R V
    inst✝⁵ : LieRingModule L V
    inst✝⁴ : LieModule R L V
    inst✝³ : LieRingModule A V
    inst✝² : LieModule R A V
    inst✝¹ : IsLieTower L A V
    inst✝ : IsLieTower A L V
    χ : A → R
    z : L
    v : V
    hv : ∀ (x : A), Eq (Bracket.bracket x v) (HSMul.hSMul (χ x) v)
    a : A
    hv' : Ne v 0
    U' : OrderHom Nat (Submodule R V) := { toFun := fun n => Submodule.span R (set …
    map_U'_le : ∀ (n : Nat), LE.le (Submodule.map ((LieModule.toEnd R L V) z) (U'  …
    T_apply_succ : ∀ (w : A) (n : Nat), LE.le (Submodule.map (LieModule.T χ w) (U' …
    U : LieSubmodule R A V := { toSubmodule := iSup fun k => U' k, lie_mem := ⋯ }
    hzU : ∀ (x : V), Membership.mem U x → Membership.mem U (((LieModule.toEnd R L  …
    trace_za_zero : Eq ((LinearMap.trace R (Subtype fun x => Membership.mem U x))  …
    trace_T_U_zero : ∀ (w : A), Eq ((LinearMap.trace R (Subtype fun x => Membershi …
    trace_za : Eq ((LinearMap.trace R (Subtype fun x => Membership.mem U x)) ((Lie …
    ⊢ Eq (χ (Bracket.bracket z a)) 0
  -/
  suffices finrank R U ≠ 0 by simp_all
  /-
    case inr
    R : Type u_1
    L : Type u_2
    A : Type u_3
    V : Type u_4
    inst✝¹⁹ : CommRing R
    inst✝¹⁸ : IsPrincipalIdealRing R
    inst✝¹⁷ : IsDomain R
    inst✝¹⁶ : CharZero R
    inst✝¹⁵ : LieRing L
    inst✝¹⁴ : LieAlgebra R L
    inst✝¹³ : LieRing A
    inst✝¹² : LieAlgebra R A
    inst✝¹¹ : Bracket L A
    inst✝¹⁰ : Bracket A L
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : Module R V
    inst✝⁷ : Module.Free R V
    inst✝⁶ : Module.Finite R V
    inst✝⁵ : LieRingModule L V
    inst✝⁴ : LieModule R L V
    inst✝³ : LieRingModule A V
    inst✝² : LieModule R A V
    inst✝¹ : IsLieTower L A V
    inst✝ : IsLieTower A L V
    χ : A → R
    z : L
    v : V
    hv : ∀ (x : A), Eq (Bracket.bracket x v) (HSMul.hSMul (χ x) v)
    a : A
    hv' : Ne v 0
    U' : OrderHom Nat (Submodule R V) := { toFun := fun n => Submodule.span R (set …
    map_U'_le : ∀ (n : Nat), LE.le (Submodule.map ((LieModule.toEnd R L V) z) (U'  …
    T_apply_succ : ∀ (w : A) (n : Nat), LE.le (Submodule.map (LieModule.T χ w) (U' …
    U : LieSubmodule R A V := { toSubmodule := iSup fun k => U' k, lie_mem := ⋯ }
    hzU : ∀ (x : V), Membership.mem U x → Membership.mem U (((LieModule.toEnd R L  …
    trace_za_zero : Eq ((LinearMap.trace R (Subtype fun x => Membership.mem U x))  …
    trace_T_U_zero : ∀ (w : A), Eq ((LinearMap.trace R (Subtype fun x => Membershi …
    trace_za : Eq ((LinearMap.trace R (Subtype fun x => Membership.mem U x)) ((Lie …
    ⊢ Ne (Module.finrank R (Subtype fun x => Membership.mem U x)) 0
  -/
  suffices Nontrivial U from Module.finrank_pos.ne'
  have hvU : v ∈ U := by
    apply Submodule.mem_iSup_of_mem 1
    apply Submodule.subset_span
    use 0, zero_lt_one
    rw [pow_zero, LinearMap.one_apply]
  /-
    case inr
    R : Type u_1
    L : Type u_2
    A : Type u_3
    V : Type u_4
    inst✝¹⁹ : CommRing R
    inst✝¹⁸ : IsPrincipalIdealRing R
    inst✝¹⁷ : IsDomain R
    inst✝¹⁶ : CharZero R
    inst✝¹⁵ : LieRing L
    inst✝¹⁴ : LieAlgebra R L
    inst✝¹³ : LieRing A
    inst✝¹² : LieAlgebra R A
    inst✝¹¹ : Bracket L A
    inst✝¹⁰ : Bracket A L
    inst✝⁹ : AddCommGroup V
    inst✝⁸ : Module R V
    inst✝⁷ : Module.Free R V
    inst✝⁶ : Module.Finite R V
    inst✝⁵ : LieRingModule L V
    inst✝⁴ : LieModule R L V
    inst✝³ : LieRingModule A V
    inst✝² : LieModule R A V
    inst✝¹ : IsLieTower L A V
    inst✝ : IsLieTower A L V
    χ : A → R
    z : L
    v : V
    hv : ∀ (x : A), Eq (Bracket.bracket x v) (HSMul.hSMul (χ x) v)
    a : A
    hv' : Ne v 0
    U' : OrderHom Nat (Submodule R V) := { toFun := fun n => Submodule.span R (set …
    map_U'_le : ∀ (n : Nat), LE.le (Submodule.map ((LieModule.toEnd R L V) z) (U'  …
    T_apply_succ : ∀ (w : A) (n : Nat), LE.le (Submodule.map (LieModule.T χ w) (U' …
    U : LieSubmodule R A V := { toSubmodule := iSup fun k => U' k, lie_mem := ⋯ }
    hzU : ∀ (x : V), Membership.mem U x → Membership.mem U (((LieModule.toEnd R L  …
    trace_za_zero : Eq ((LinearMap.trace R (Subtype fun x => Membership.mem U x))  …
    trace_T_U_zero : ∀ (w : A), Eq ((LinearMap.trace R (Subtype fun x => Membershi …
    trace_za : Eq ((LinearMap.trace R (Subtype fun x => Membership.mem U x)) ((Lie …
    hvU : Membership.mem U v
    ⊢ Nontrivial (Subtype fun x => Membership.mem U x)
  -/
  exact nontrivial_of_ne ⟨v, hvU⟩ 0 <| by simp [hv']
  /-
    🎉 no goals
  -/


variable (R V) in
/-- The weight space of `V` with respect to `χ : A → R`, a priori a Lie submodule for `A`, is also a
Lie submodule for `L`. -/
def weightSpaceOfIsLieTower (χ : A → R) : LieSubmodule R L V :=
  { toSubmodule := weightSpace V χ
    lie_mem {z v} hv := weightSpaceOfIsLieTower_aux χ z v hv }


open Submodule in
theorem exists_nontrivial_weightSpace_of_lieIdeal [LieModule.IsTriangularizable k L V]
    (A : LieIdeal k L) (hA : IsCoatom A.toSubmodule)
    (χ₀ : Module.Dual k A) [Nontrivial (weightSpace V χ₀)] :
    ∃ (χ : Module.Dual k L), Nontrivial (weightSpace V χ) := by
  /-
    k : Type u_1
    inst✝¹⁰ : Field k
    L : Type u_2
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra k L
    V : Type u_3
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module k V
    inst✝⁵ : LieRingModule L V
    inst✝⁴ : LieModule k L V
    inst✝³ : CharZero k
    inst✝² : Module.Finite k V
    inst✝¹ : LieModule.IsTriangularizable k L V
    A : LieIdeal k L
    hA : IsCoatom ↑A
    χ₀ : Module.Dual k (Subtype fun x => Membership.mem A x)
    inst✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑ …
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  obtain ⟨z, -, hz⟩ := SetLike.exists_of_lt (hA.lt_top)
  /-
    case intro.intro
    k : Type u_1
    inst✝¹⁰ : Field k
    L : Type u_2
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra k L
    V : Type u_3
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module k V
    inst✝⁵ : LieRingModule L V
    inst✝⁴ : LieModule k L V
    inst✝³ : CharZero k
    inst✝² : Module.Finite k V
    inst✝¹ : LieModule.IsTriangularizable k L V
    A : LieIdeal k L
    hA : IsCoatom ↑A
    χ₀ : Module.Dual k (Subtype fun x => Membership.mem A x)
    inst✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑ …
    z : L
    hz : Not (Membership.mem (↑A) z)
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  let e : (k ∙ z) ≃ₗ[k] k := (LinearEquiv.toSpanNonzeroSingleton k L z <| by aesop).symm
  /-
    case intro.intro
    k : Type u_1
    inst✝¹⁰ : Field k
    L : Type u_2
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra k L
    V : Type u_3
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module k V
    inst✝⁵ : LieRingModule L V
    inst✝⁴ : LieModule k L V
    inst✝³ : CharZero k
    inst✝² : Module.Finite k V
    inst✝¹ : LieModule.IsTriangularizable k L V
    A : LieIdeal k L
    hA : IsCoatom ↑A
    χ₀ : Module.Dual k (Subtype fun x => Membership.mem A x)
    inst✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑ …
    z : L
    hz : Not (Membership.mem (↑A) z)
    e : LinearEquiv (RingHom.id k) (Subtype fun x => Membership.mem (Submodule.spa …
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  have he : ∀ x, e x • z = x := by simp [e]
  /-
    case intro.intro
    k : Type u_1
    inst✝¹⁰ : Field k
    L : Type u_2
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra k L
    V : Type u_3
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module k V
    inst✝⁵ : LieRingModule L V
    inst✝⁴ : LieModule k L V
    inst✝³ : CharZero k
    inst✝² : Module.Finite k V
    inst✝¹ : LieModule.IsTriangularizable k L V
    A : LieIdeal k L
    hA : IsCoatom ↑A
    χ₀ : Module.Dual k (Subtype fun x => Membership.mem A x)
    inst✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑ …
    z : L
    hz : Not (Membership.mem (↑A) z)
    e : LinearEquiv (RingHom.id k) (Subtype fun x => Membership.mem (Submodule.spa …
    he : ∀ (x : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singl …
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  have hA : IsCompl A.toSubmodule (k ∙ z) := isCompl_span_singleton_of_isCoatom_of_not_mem hA hz
  /-
    case intro.intro
    k : Type u_1
    inst✝¹⁰ : Field k
    L : Type u_2
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra k L
    V : Type u_3
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module k V
    inst✝⁵ : LieRingModule L V
    inst✝⁴ : LieModule k L V
    inst✝³ : CharZero k
    inst✝² : Module.Finite k V
    inst✝¹ : LieModule.IsTriangularizable k L V
    A : LieIdeal k L
    hA✝ : IsCoatom ↑A
    χ₀ : Module.Dual k (Subtype fun x => Membership.mem A x)
    inst✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑ …
    z : L
    hz : Not (Membership.mem (↑A) z)
    e : LinearEquiv (RingHom.id k) (Subtype fun x => Membership.mem (Submodule.spa …
    he : ∀ (x : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singl …
    hA : IsCompl (↑A) (Submodule.span k (Singleton.singleton z))
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  let π₁ : L →ₗ[k] A       := A.toSubmodule.linearProjOfIsCompl (k ∙ z) hA
  /-
    case intro.intro
    k : Type u_1
    inst✝¹⁰ : Field k
    L : Type u_2
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra k L
    V : Type u_3
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module k V
    inst✝⁵ : LieRingModule L V
    inst✝⁴ : LieModule k L V
    inst✝³ : CharZero k
    inst✝² : Module.Finite k V
    inst✝¹ : LieModule.IsTriangularizable k L V
    A : LieIdeal k L
    hA✝ : IsCoatom ↑A
    χ₀ : Module.Dual k (Subtype fun x => Membership.mem A x)
    inst✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑ …
    z : L
    hz : Not (Membership.mem (↑A) z)
    e : LinearEquiv (RingHom.id k) (Subtype fun x => Membership.mem (Submodule.spa …
    he : ∀ (x : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singl …
    hA : IsCompl (↑A) (Submodule.span k (Singleton.singleton z))
    π₁ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem A x) := (↑A). …
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  let π₂ : L →ₗ[k] (k ∙ z) := (k ∙ z).linearProjOfIsCompl ↑A hA.symm

  /-
    case intro.intro
    k : Type u_1
    inst✝¹⁰ : Field k
    L : Type u_2
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra k L
    V : Type u_3
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module k V
    inst✝⁵ : LieRingModule L V
    inst✝⁴ : LieModule k L V
    inst✝³ : CharZero k
    inst✝² : Module.Finite k V
    inst✝¹ : LieModule.IsTriangularizable k L V
    A : LieIdeal k L
    hA✝ : IsCoatom ↑A
    χ₀ : Module.Dual k (Subtype fun x => Membership.mem A x)
    inst✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑ …
    z : L
    hz : Not (Membership.mem (↑A) z)
    e : LinearEquiv (RingHom.id k) (Subtype fun x => Membership.mem (Submodule.spa …
    he : ∀ (x : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singl …
    hA : IsCompl (↑A) (Submodule.span k (Singleton.singleton z))
    π₁ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem A x) := (↑A). …
    π₂ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem (Submodule.sp …
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  set W : LieSubmodule k L V := weightSpaceOfIsLieTower k V χ₀
  obtain ⟨c, hc⟩ : ∃ c, (toEnd k _ W z).HasEigenvalue c := by
    have : Nontrivial W := inferInstanceAs (Nontrivial (weightSpace V χ₀))
    apply Module.End.exists_hasEigenvalue_of_genEigenspace_eq_top
    exact LieModule.IsTriangularizable.maxGenEigenspace_eq_top z

  /-
    case intro.intro.intro
    k : Type u_1
    inst✝¹⁰ : Field k
    L : Type u_2
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra k L
    V : Type u_3
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module k V
    inst✝⁵ : LieRingModule L V
    inst✝⁴ : LieModule k L V
    inst✝³ : CharZero k
    inst✝² : Module.Finite k V
    inst✝¹ : LieModule.IsTriangularizable k L V
    A : LieIdeal k L
    hA✝ : IsCoatom ↑A
    χ₀ : Module.Dual k (Subtype fun x => Membership.mem A x)
    inst✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑ …
    z : L
    hz : Not (Membership.mem (↑A) z)
    e : LinearEquiv (RingHom.id k) (Subtype fun x => Membership.mem (Submodule.spa …
    he : ∀ (x : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singl …
    hA : IsCompl (↑A) (Submodule.span k (Singleton.singleton z))
    π₁ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem A x) := (↑A). …
    π₂ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem (Submodule.sp …
    W : LieSubmodule k L V := LieModule.weightSpaceOfIsLieTower k V ⇑χ₀
    c : k
    hc : ((LieModule.toEnd k L (Subtype fun x => Membership.mem W x)) z).HasEigenv …
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  obtain ⟨⟨v, hv⟩, hvc⟩ := hc.exists_hasEigenvector
  have hv' : ∀ (x : ↥A), ⁅x, v⁆ = χ₀ x • v := by
    simpa [W, weightSpaceOfIsLieTower, mem_weightSpace] using hv

  /-
    case intro.intro.intro.intro.mk
    k : Type u_1
    inst✝¹⁰ : Field k
    L : Type u_2
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra k L
    V : Type u_3
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module k V
    inst✝⁵ : LieRingModule L V
    inst✝⁴ : LieModule k L V
    inst✝³ : CharZero k
    inst✝² : Module.Finite k V
    inst✝¹ : LieModule.IsTriangularizable k L V
    A : LieIdeal k L
    hA✝ : IsCoatom ↑A
    χ₀ : Module.Dual k (Subtype fun x => Membership.mem A x)
    inst✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑ …
    z : L
    hz : Not (Membership.mem (↑A) z)
    e : LinearEquiv (RingHom.id k) (Subtype fun x => Membership.mem (Submodule.spa …
    he : ∀ (x : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singl …
    hA : IsCompl (↑A) (Submodule.span k (Singleton.singleton z))
    π₁ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem A x) := (↑A). …
    π₂ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem (Submodule.sp …
    W : LieSubmodule k L V := LieModule.weightSpaceOfIsLieTower k V ⇑χ₀
    c : k
    hc : ((LieModule.toEnd k L (Subtype fun x => Membership.mem W x)) z).HasEigenv …
    v : V
    hv : Membership.mem W v
    hvc : ((LieModule.toEnd k L (Subtype fun x => Membership.mem W x)) z).HasEigen …
    hv' : ∀ (x : Subtype fun x => Membership.mem A x), Eq (Bracket.bracket x v) (H …
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  use (χ₀.comp π₁) + c • (e.comp π₂)
  /-
    case h
    k : Type u_1
    inst✝¹⁰ : Field k
    L : Type u_2
    inst✝⁹ : LieRing L
    inst✝⁸ : LieAlgebra k L
    V : Type u_3
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module k V
    inst✝⁵ : LieRingModule L V
    inst✝⁴ : LieModule k L V
    inst✝³ : CharZero k
    inst✝² : Module.Finite k V
    inst✝¹ : LieModule.IsTriangularizable k L V
    A : LieIdeal k L
    hA✝ : IsCoatom ↑A
    χ₀ : Module.Dual k (Subtype fun x => Membership.mem A x)
    inst✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑ …
    z : L
    hz : Not (Membership.mem (↑A) z)
    e : LinearEquiv (RingHom.id k) (Subtype fun x => Membership.mem (Submodule.spa …
    he : ∀ (x : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singl …
    hA : IsCompl (↑A) (Submodule.span k (Singleton.singleton z))
    π₁ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem A x) := (↑A). …
    π₂ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem (Submodule.sp …
    W : LieSubmodule k L V := LieModule.weightSpaceOfIsLieTower k V ⇑χ₀
    c : k
    hc : ((LieModule.toEnd k L (Subtype fun x => Membership.mem W x)) z).HasEigenv …
    v : V
    hv : Membership.mem W v
    hvc : ((LieModule.toEnd k L (Subtype fun x => Membership.mem W x)) z).HasEigen …
    hv' : ∀ (x : Subtype fun x => Membership.mem A x), Eq (Bracket.bracket x v) (H …
    ⊢ Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑(HAdd. …
  -/
  refine nontrivial_of_ne ⟨v, ?_⟩ 0 ?_
    /-
      case h.refine_1
      k : Type u_1
      inst✝¹⁰ : Field k
      L : Type u_2
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra k L
      V : Type u_3
      inst✝⁷ : AddCommGroup V
      inst✝⁶ : Module k V
      inst✝⁵ : LieRingModule L V
      inst✝⁴ : LieModule k L V
      inst✝³ : CharZero k
      inst✝² : Module.Finite k V
      inst✝¹ : LieModule.IsTriangularizable k L V
      A : LieIdeal k L
      hA✝ : IsCoatom ↑A
      χ₀ : Module.Dual k (Subtype fun x => Membership.mem A x)
      inst✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑ …
      z : L
      hz : Not (Membership.mem (↑A) z)
      e : LinearEquiv (RingHom.id k) (Subtype fun x => Membership.mem (Submodule.spa …
      he : ∀ (x : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singl …
      hA : IsCompl (↑A) (Submodule.span k (Singleton.singleton z))
      π₁ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem A x) := (↑A). …
      π₂ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem (Submodule.sp …
      W : LieSubmodule k L V := LieModule.weightSpaceOfIsLieTower k V ⇑χ₀
      c : k
      hc : ((LieModule.toEnd k L (Subtype fun x => Membership.mem W x)) z).HasEigenv …
      v : V
      hv : Membership.mem W v
      hvc : ((LieModule.toEnd k L (Subtype fun x => Membership.mem W x)) z).HasEigen …
      hv' : ∀ (x : Subtype fun x => Membership.mem A x), Eq (Bracket.bracket x v) (H …
      ⊢ Membership.mem (LieModule.weightSpace V ⇑(HAdd.hAdd (LinearMap.comp χ₀ π₁) ( …
    -/
  · rw [mem_weightSpace]
    /-
      case h.refine_1
      k : Type u_1
      inst✝¹⁰ : Field k
      L : Type u_2
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra k L
      V : Type u_3
      inst✝⁷ : AddCommGroup V
      inst✝⁶ : Module k V
      inst✝⁵ : LieRingModule L V
      inst✝⁴ : LieModule k L V
      inst✝³ : CharZero k
      inst✝² : Module.Finite k V
      inst✝¹ : LieModule.IsTriangularizable k L V
      A : LieIdeal k L
      hA✝ : IsCoatom ↑A
      χ₀ : Module.Dual k (Subtype fun x => Membership.mem A x)
      inst✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑ …
      z : L
      hz : Not (Membership.mem (↑A) z)
      e : LinearEquiv (RingHom.id k) (Subtype fun x => Membership.mem (Submodule.spa …
      he : ∀ (x : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singl …
      hA : IsCompl (↑A) (Submodule.span k (Singleton.singleton z))
      π₁ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem A x) := (↑A). …
      π₂ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem (Submodule.sp …
      W : LieSubmodule k L V := LieModule.weightSpaceOfIsLieTower k V ⇑χ₀
      c : k
      hc : ((LieModule.toEnd k L (Subtype fun x => Membership.mem W x)) z).HasEigenv …
      v : V
      hv : Membership.mem W v
      hvc : ((LieModule.toEnd k L (Subtype fun x => Membership.mem W x)) z).HasEigen …
      hv' : ∀ (x : Subtype fun x => Membership.mem A x), Eq (Bracket.bracket x v) (H …
      ⊢ ∀ (x : L), Eq (Bracket.bracket x v) (HSMul.hSMul ((HAdd.hAdd (LinearMap.comp …
    -/
    intro x
    /-
      case h.refine_1
      k : Type u_1
      inst✝¹⁰ : Field k
      L : Type u_2
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra k L
      V : Type u_3
      inst✝⁷ : AddCommGroup V
      inst✝⁶ : Module k V
      inst✝⁵ : LieRingModule L V
      inst✝⁴ : LieModule k L V
      inst✝³ : CharZero k
      inst✝² : Module.Finite k V
      inst✝¹ : LieModule.IsTriangularizable k L V
      A : LieIdeal k L
      hA✝ : IsCoatom ↑A
      χ₀ : Module.Dual k (Subtype fun x => Membership.mem A x)
      inst✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑ …
      z : L
      hz : Not (Membership.mem (↑A) z)
      e : LinearEquiv (RingHom.id k) (Subtype fun x => Membership.mem (Submodule.spa …
      he : ∀ (x : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singl …
      hA : IsCompl (↑A) (Submodule.span k (Singleton.singleton z))
      π₁ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem A x) := (↑A). …
      π₂ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem (Submodule.sp …
      W : LieSubmodule k L V := LieModule.weightSpaceOfIsLieTower k V ⇑χ₀
      c : k
      hc : ((LieModule.toEnd k L (Subtype fun x => Membership.mem W x)) z).HasEigenv …
      v : V
      hv : Membership.mem W v
      hvc : ((LieModule.toEnd k L (Subtype fun x => Membership.mem W x)) z).HasEigen …
      hv' : ∀ (x : Subtype fun x => Membership.mem A x), Eq (Bracket.bracket x v) (H …
      x : L
      ⊢ Eq (Bracket.bracket x v) (HSMul.hSMul ((HAdd.hAdd (LinearMap.comp χ₀ π₁) (HS …
    -/
    have hπ : (π₁ x : L) + π₂ x = x := linear_proj_add_linearProjOfIsCompl_eq_self hA x
    suffices ⁅(π₂ x : L), v⁆ = (c • e (π₂ x)) • v by
      calc ⁅x, v⁆
          = ⁅π₁ x, v⁆       + ⁅(π₂ x : L), v⁆    := congr(⁅$hπ.symm, v⁆) ▸ add_lie _ _ _
        _ =  χ₀ (π₁ x) • v  + (c • e (π₂ x)) • v := by rw [hv' (π₁ x), this]
        _ = _ := by simp [add_smul]
    calc ⁅(π₂ x : L), v⁆
        = e (π₂ x) • ↑(c • ⟨v, hv⟩ : W) := by rw [← he, smul_lie, ← hvc.apply_eq_smul]; rfl
      _ = (c • e (π₂ x)) • v              := by rw [smul_assoc, smul_comm]; rfl
    /-
      case h.refine_2
      k : Type u_1
      inst✝¹⁰ : Field k
      L : Type u_2
      inst✝⁹ : LieRing L
      inst✝⁸ : LieAlgebra k L
      V : Type u_3
      inst✝⁷ : AddCommGroup V
      inst✝⁶ : Module k V
      inst✝⁵ : LieRingModule L V
      inst✝⁴ : LieModule k L V
      inst✝³ : CharZero k
      inst✝² : Module.Finite k V
      inst✝¹ : LieModule.IsTriangularizable k L V
      A : LieIdeal k L
      hA✝ : IsCoatom ↑A
      χ₀ : Module.Dual k (Subtype fun x => Membership.mem A x)
      inst✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑ …
      z : L
      hz : Not (Membership.mem (↑A) z)
      e : LinearEquiv (RingHom.id k) (Subtype fun x => Membership.mem (Submodule.spa …
      he : ∀ (x : Subtype fun x => Membership.mem (Submodule.span k (Singleton.singl …
      hA : IsCompl (↑A) (Submodule.span k (Singleton.singleton z))
      π₁ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem A x) := (↑A). …
      π₂ : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem (Submodule.sp …
      W : LieSubmodule k L V := LieModule.weightSpaceOfIsLieTower k V ⇑χ₀
      c : k
      hc : ((LieModule.toEnd k L (Subtype fun x => Membership.mem W x)) z).HasEigenv …
      v : V
      hv : Membership.mem W v
      hvc : ((LieModule.toEnd k L (Subtype fun x => Membership.mem W x)) z).HasEigen …
      hv' : ∀ (x : Subtype fun x => Membership.mem A x), Eq (Bracket.bracket x v) (H …
      ⊢ Ne ⟨v, ⋯⟩ 0
    -/
  · simpa [ne_eq, LieSubmodule.mk_eq_zero] using hvc.right
    /-
      🎉 no goals
    -/


private lemma exists_forall_lie_eq_smul_of_isSolvable_of_finite
    (L : Type*) [LieRing L] [LieAlgebra k L] [LieRingModule L V] [LieModule k L V]
    [IsSolvable k L] [LieModule.IsTriangularizable k L V] [Module.Finite k L] :
    ∃ χ : Module.Dual k L, Nontrivial (weightSpace V χ) := by
  /-
    k : Type u_1
    inst✝¹² : Field k
    V : Type u_3
    inst✝¹¹ : AddCommGroup V
    inst✝¹⁰ : Module k V
    inst✝⁹ : CharZero k
    inst✝⁸ : Module.Finite k V
    inst✝⁷ : Nontrivial V
    L : Type u_4
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra k L
    inst✝⁴ : LieRingModule L V
    inst✝³ : LieModule k L V
    inst✝² : LieAlgebra.IsSolvable k L
    inst✝¹ : LieModule.IsTriangularizable k L V
    inst✝ : Module.Finite k L
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  obtain H|⟨A, hA, hAL⟩ := eq_top_or_exists_le_coatom (derivedSeries k L 1).toSubmodule
    /-
      case inl
      k : Type u_1
      inst✝¹² : Field k
      V : Type u_3
      inst✝¹¹ : AddCommGroup V
      inst✝¹⁰ : Module k V
      inst✝⁹ : CharZero k
      inst✝⁸ : Module.Finite k V
      inst✝⁷ : Nontrivial V
      L : Type u_4
      inst✝⁶ : LieRing L
      inst✝⁵ : LieAlgebra k L
      inst✝⁴ : LieRingModule L V
      inst✝³ : LieModule k L V
      inst✝² : LieAlgebra.IsSolvable k L
      inst✝¹ : LieModule.IsTriangularizable k L V
      inst✝ : Module.Finite k L
      H : Eq (↑(LieAlgebra.derivedSeries k L 1)) Top.top
      ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
    -/
  · obtain _|_ := subsingleton_or_nontrivial L
      /-
        case inl.inl
        k : Type u_1
        inst✝¹² : Field k
        V : Type u_3
        inst✝¹¹ : AddCommGroup V
        inst✝¹⁰ : Module k V
        inst✝⁹ : CharZero k
        inst✝⁸ : Module.Finite k V
        inst✝⁷ : Nontrivial V
        L : Type u_4
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra k L
        inst✝⁴ : LieRingModule L V
        inst✝³ : LieModule k L V
        inst✝² : LieAlgebra.IsSolvable k L
        inst✝¹ : LieModule.IsTriangularizable k L V
        inst✝ : Module.Finite k L
        H : Eq (↑(LieAlgebra.derivedSeries k L 1)) Top.top
        h✝ : Subsingleton L
        ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
      -/
    · use 0
      /-
        case h
        k : Type u_1
        inst✝¹² : Field k
        V : Type u_3
        inst✝¹¹ : AddCommGroup V
        inst✝¹⁰ : Module k V
        inst✝⁹ : CharZero k
        inst✝⁸ : Module.Finite k V
        inst✝⁷ : Nontrivial V
        L : Type u_4
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra k L
        inst✝⁴ : LieRingModule L V
        inst✝³ : LieModule k L V
        inst✝² : LieAlgebra.IsSolvable k L
        inst✝¹ : LieModule.IsTriangularizable k L V
        inst✝ : Module.Finite k L
        H : Eq (↑(LieAlgebra.derivedSeries k L 1)) Top.top
        h✝ : Subsingleton L
        ⊢ Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑0) x)
      -/
      simpa [mem_weightSpace, nontrivial_iff] using exists_pair_ne V
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        k : Type u_1
        inst✝¹² : Field k
        V : Type u_3
        inst✝¹¹ : AddCommGroup V
        inst✝¹⁰ : Module k V
        inst✝⁹ : CharZero k
        inst✝⁸ : Module.Finite k V
        inst✝⁷ : Nontrivial V
        L : Type u_4
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra k L
        inst✝⁴ : LieRingModule L V
        inst✝³ : LieModule k L V
        inst✝² : LieAlgebra.IsSolvable k L
        inst✝¹ : LieModule.IsTriangularizable k L V
        inst✝ : Module.Finite k L
        H : Eq (↑(LieAlgebra.derivedSeries k L 1)) Top.top
        h✝ : Nontrivial L
        ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
      -/
    · rw [LieSubmodule.toSubmodule_eq_top] at H
      /-
        case inl.inr
        k : Type u_1
        inst✝¹² : Field k
        V : Type u_3
        inst✝¹¹ : AddCommGroup V
        inst✝¹⁰ : Module k V
        inst✝⁹ : CharZero k
        inst✝⁸ : Module.Finite k V
        inst✝⁷ : Nontrivial V
        L : Type u_4
        inst✝⁶ : LieRing L
        inst✝⁵ : LieAlgebra k L
        inst✝⁴ : LieRingModule L V
        inst✝³ : LieModule k L V
        inst✝² : LieAlgebra.IsSolvable k L
        inst✝¹ : LieModule.IsTriangularizable k L V
        inst✝ : Module.Finite k L
        H : Eq (LieAlgebra.derivedSeries k L 1) Top.top
        h✝ : Nontrivial L
        ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
      -/
      exact ((derivedSeries_lt_top_of_solvable k L).ne H).elim
      /-
        🎉 no goals
      -/
  /-
    case inr.intro.intro
    k : Type u_1
    inst✝¹² : Field k
    V : Type u_3
    inst✝¹¹ : AddCommGroup V
    inst✝¹⁰ : Module k V
    inst✝⁹ : CharZero k
    inst✝⁸ : Module.Finite k V
    inst✝⁷ : Nontrivial V
    L : Type u_4
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra k L
    inst✝⁴ : LieRingModule L V
    inst✝³ : LieModule k L V
    inst✝² : LieAlgebra.IsSolvable k L
    inst✝¹ : LieModule.IsTriangularizable k L V
    inst✝ : Module.Finite k L
    A : Submodule k L
    hA : IsCoatom A
    hAL : LE.le (↑(LieAlgebra.derivedSeries k L 1)) A
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  lift A to LieIdeal k L
    /-
      k : Type u_1
      inst✝¹² : Field k
      V : Type u_3
      inst✝¹¹ : AddCommGroup V
      inst✝¹⁰ : Module k V
      inst✝⁹ : CharZero k
      inst✝⁸ : Module.Finite k V
      inst✝⁷ : Nontrivial V
      L : Type u_4
      inst✝⁶ : LieRing L
      inst✝⁵ : LieAlgebra k L
      inst✝⁴ : LieRingModule L V
      inst✝³ : LieModule k L V
      inst✝² : LieAlgebra.IsSolvable k L
      inst✝¹ : LieModule.IsTriangularizable k L V
      inst✝ : Module.Finite k L
      A : Submodule k L
      hA : IsCoatom A
      hAL : LE.le (↑(LieAlgebra.derivedSeries k L 1)) A
      ⊢ ∀ {x m : L}, Membership.mem A m → Membership.mem A (Bracket.bracket x m)
    -/
  · intros
    /-
      k : Type u_1
      inst✝¹² : Field k
      V : Type u_3
      inst✝¹¹ : AddCommGroup V
      inst✝¹⁰ : Module k V
      inst✝⁹ : CharZero k
      inst✝⁸ : Module.Finite k V
      inst✝⁷ : Nontrivial V
      L : Type u_4
      inst✝⁶ : LieRing L
      inst✝⁵ : LieAlgebra k L
      inst✝⁴ : LieRingModule L V
      inst✝³ : LieModule k L V
      inst✝² : LieAlgebra.IsSolvable k L
      inst✝¹ : LieModule.IsTriangularizable k L V
      inst✝ : Module.Finite k L
      A : Submodule k L
      hA : IsCoatom A
      hAL : LE.le (↑(LieAlgebra.derivedSeries k L 1)) A
      x✝ m✝ : L
      a✝ : Membership.mem A m✝
      ⊢ Membership.mem A (Bracket.bracket x✝ m✝)
    -/
    exact hAL <| LieSubmodule.lie_mem_lie (LieSubmodule.mem_top _) (LieSubmodule.mem_top _)
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.intro.intro
    k : Type u_1
    inst✝¹² : Field k
    V : Type u_3
    inst✝¹¹ : AddCommGroup V
    inst✝¹⁰ : Module k V
    inst✝⁹ : CharZero k
    inst✝⁸ : Module.Finite k V
    inst✝⁷ : Nontrivial V
    L : Type u_4
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra k L
    inst✝⁴ : LieRingModule L V
    inst✝³ : LieModule k L V
    inst✝² : LieAlgebra.IsSolvable k L
    inst✝¹ : LieModule.IsTriangularizable k L V
    inst✝ : Module.Finite k L
    A : LieSubmodule k L L
    hA : IsCoatom ↑A
    hAL : LE.le ↑(LieAlgebra.derivedSeries k L 1) ↑A
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  change LieIdeal k L at A -- remove this line when bug in `lift` is fixed (#15865)
  /-
    case inr.intro.intro.intro
    k : Type u_1
    inst✝¹² : Field k
    V : Type u_3
    inst✝¹¹ : AddCommGroup V
    inst✝¹⁰ : Module k V
    inst✝⁹ : CharZero k
    inst✝⁸ : Module.Finite k V
    inst✝⁷ : Nontrivial V
    L : Type u_4
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra k L
    inst✝⁴ : LieRingModule L V
    inst✝³ : LieModule k L V
    inst✝² : LieAlgebra.IsSolvable k L
    inst✝¹ : LieModule.IsTriangularizable k L V
    inst✝ : Module.Finite k L
    A : LieIdeal k L
    hA : IsCoatom ↑A
    hAL : LE.le ↑(LieAlgebra.derivedSeries k L 1) ↑A
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  obtain ⟨χ', _⟩ := exists_forall_lie_eq_smul_of_isSolvable_of_finite A
  /-
    case inr.intro.intro.intro.intro
    k : Type u_1
    inst✝¹² : Field k
    V : Type u_3
    inst✝¹¹ : AddCommGroup V
    inst✝¹⁰ : Module k V
    inst✝⁹ : CharZero k
    inst✝⁸ : Module.Finite k V
    inst✝⁷ : Nontrivial V
    L : Type u_4
    inst✝⁶ : LieRing L
    inst✝⁵ : LieAlgebra k L
    inst✝⁴ : LieRingModule L V
    inst✝³ : LieModule k L V
    inst✝² : LieAlgebra.IsSolvable k L
    inst✝¹ : LieModule.IsTriangularizable k L V
    inst✝ : Module.Finite k L
    A : LieIdeal k L
    hA : IsCoatom ↑A
    hAL : LE.le ↑(LieAlgebra.derivedSeries k L 1) ↑A
    χ' : Module.Dual k (Subtype fun x => Membership.mem A x)
    h✝ : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑χ') …
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  exact exists_nontrivial_weightSpace_of_lieIdeal A hA χ'
  /-
    🎉 no goals
  -/
termination_by Module.finrank k L
decreasing_by
  simp_wf
  rw [← finrank_top k L]
  apply Submodule.finrank_lt_finrank_of_lt
  exact hA.lt_top


/-- **Lie's theorem**: Lie modules of solvable Lie algebras over fields of characteristic 0
have a common eigenvector for the action of all elements of the Lie algebra.

See `LieModule.exists_nontrivial_weightSpace_of_isNilpotent` for the variant that
assumes that `L` is nilpotent and drops the condition that `k` is of characteristic zero. -/
theorem exists_nontrivial_weightSpace_of_isSolvable
    [IsSolvable k L] [LieModule.IsTriangularizable k L V] :
    ∃ χ : Module.Dual k L, Nontrivial (weightSpace V χ) := by
  /-
    k : Type u_1
    inst✝¹¹ : Field k
    L : Type u_2
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra k L
    V : Type u_3
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : Module k V
    inst✝⁶ : LieRingModule L V
    inst✝⁵ : LieModule k L V
    inst✝⁴ : CharZero k
    inst✝³ : Module.Finite k V
    inst✝² : Nontrivial V
    inst✝¹ : LieAlgebra.IsSolvable k L
    inst✝ : LieModule.IsTriangularizable k L V
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  let imL := (toEnd k L V).range
  let toEndo : L →ₗ[k] imL := LinearMap.codRestrict imL.toSubmodule (toEnd k L V)
      (fun x ↦ LinearMap.mem_range.mpr ⟨x, rfl⟩ : ∀ x : L, (toEnd k L V) x ∈ imL)
  /-
    k : Type u_1
    inst✝¹¹ : Field k
    L : Type u_2
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra k L
    V : Type u_3
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : Module k V
    inst✝⁶ : LieRingModule L V
    inst✝⁵ : LieModule k L V
    inst✝⁴ : CharZero k
    inst✝³ : Module.Finite k V
    inst✝² : Nontrivial V
    inst✝¹ : LieAlgebra.IsSolvable k L
    inst✝ : LieModule.IsTriangularizable k L V
    imL : LieSubalgebra k (Module.End k V) := (LieModule.toEnd k L V).range
    toEndo : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem imL x) := …
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  have ⟨χ, h⟩ := exists_forall_lie_eq_smul_of_isSolvable_of_finite k V imL
  /-
    k : Type u_1
    inst✝¹¹ : Field k
    L : Type u_2
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra k L
    V : Type u_3
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : Module k V
    inst✝⁶ : LieRingModule L V
    inst✝⁵ : LieModule k L V
    inst✝⁴ : CharZero k
    inst✝³ : Module.Finite k V
    inst✝² : Nontrivial V
    inst✝¹ : LieAlgebra.IsSolvable k L
    inst✝ : LieModule.IsTriangularizable k L V
    imL : LieSubalgebra k (Module.End k V) := (LieModule.toEnd k L V).range
    toEndo : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem imL x) := …
    χ : Module.Dual k (Subtype fun x => Membership.mem imL x)
    h : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑χ) x)
    ⊢ Exists fun χ => Nontrivial (Subtype fun x => Membership.mem (LieModule.weigh …
  -/
  use χ.comp toEndo
  /-
    case h
    k : Type u_1
    inst✝¹¹ : Field k
    L : Type u_2
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra k L
    V : Type u_3
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : Module k V
    inst✝⁶ : LieRingModule L V
    inst✝⁵ : LieModule k L V
    inst✝⁴ : CharZero k
    inst✝³ : Module.Finite k V
    inst✝² : Nontrivial V
    inst✝¹ : LieAlgebra.IsSolvable k L
    inst✝ : LieModule.IsTriangularizable k L V
    imL : LieSubalgebra k (Module.End k V) := (LieModule.toEnd k L V).range
    toEndo : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem imL x) := …
    χ : Module.Dual k (Subtype fun x => Membership.mem imL x)
    h : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑χ) x)
    ⊢ Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑(Linea …
  -/
  obtain ⟨⟨v, hv⟩, hv0⟩ := exists_ne (0 : weightSpace V χ)
  /-
    case h.intro.mk
    k : Type u_1
    inst✝¹¹ : Field k
    L : Type u_2
    inst✝¹⁰ : LieRing L
    inst✝⁹ : LieAlgebra k L
    V : Type u_3
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : Module k V
    inst✝⁶ : LieRingModule L V
    inst✝⁵ : LieModule k L V
    inst✝⁴ : CharZero k
    inst✝³ : Module.Finite k V
    inst✝² : Nontrivial V
    inst✝¹ : LieAlgebra.IsSolvable k L
    inst✝ : LieModule.IsTriangularizable k L V
    imL : LieSubalgebra k (Module.End k V) := (LieModule.toEnd k L V).range
    toEndo : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem imL x) := …
    χ : Module.Dual k (Subtype fun x => Membership.mem imL x)
    h : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑χ) x)
    v : V
    hv : Membership.mem (LieModule.weightSpace V ⇑χ) v
    hv0 : Ne ⟨v, hv⟩ 0
    ⊢ Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑(Linea …
  -/
  refine nontrivial_of_ne ⟨v, ?_⟩ 0 ?_
    /-
      case h.intro.mk.refine_1
      k : Type u_1
      inst✝¹¹ : Field k
      L : Type u_2
      inst✝¹⁰ : LieRing L
      inst✝⁹ : LieAlgebra k L
      V : Type u_3
      inst✝⁸ : AddCommGroup V
      inst✝⁷ : Module k V
      inst✝⁶ : LieRingModule L V
      inst✝⁵ : LieModule k L V
      inst✝⁴ : CharZero k
      inst✝³ : Module.Finite k V
      inst✝² : Nontrivial V
      inst✝¹ : LieAlgebra.IsSolvable k L
      inst✝ : LieModule.IsTriangularizable k L V
      imL : LieSubalgebra k (Module.End k V) := (LieModule.toEnd k L V).range
      toEndo : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem imL x) := …
      χ : Module.Dual k (Subtype fun x => Membership.mem imL x)
      h : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑χ) x)
      v : V
      hv : Membership.mem (LieModule.weightSpace V ⇑χ) v
      hv0 : Ne ⟨v, hv⟩ 0
      ⊢ Membership.mem (LieModule.weightSpace V ⇑(LinearMap.comp χ toEndo)) v
    -/
  · rw [mem_weightSpace] at hv ⊢
    /-
      case h.intro.mk.refine_1
      k : Type u_1
      inst✝¹¹ : Field k
      L : Type u_2
      inst✝¹⁰ : LieRing L
      inst✝⁹ : LieAlgebra k L
      V : Type u_3
      inst✝⁸ : AddCommGroup V
      inst✝⁷ : Module k V
      inst✝⁶ : LieRingModule L V
      inst✝⁵ : LieModule k L V
      inst✝⁴ : CharZero k
      inst✝³ : Module.Finite k V
      inst✝² : Nontrivial V
      inst✝¹ : LieAlgebra.IsSolvable k L
      inst✝ : LieModule.IsTriangularizable k L V
      imL : LieSubalgebra k (Module.End k V) := (LieModule.toEnd k L V).range
      toEndo : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem imL x) := …
      χ : Module.Dual k (Subtype fun x => Membership.mem imL x)
      h : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑χ) x)
      v : V
      hv✝ : Membership.mem (LieModule.weightSpace V ⇑χ) v
      hv : ∀ (x : Subtype fun x => Membership.mem imL x), Eq (Bracket.bracket x v) ( …
      hv0 : Ne ⟨v, hv✝⟩ 0
      ⊢ ∀ (x : L), Eq (Bracket.bracket x v) (HSMul.hSMul ((LinearMap.comp χ toEndo)  …
    -/
    intro x
    /-
      case h.intro.mk.refine_1
      k : Type u_1
      inst✝¹¹ : Field k
      L : Type u_2
      inst✝¹⁰ : LieRing L
      inst✝⁹ : LieAlgebra k L
      V : Type u_3
      inst✝⁸ : AddCommGroup V
      inst✝⁷ : Module k V
      inst✝⁶ : LieRingModule L V
      inst✝⁵ : LieModule k L V
      inst✝⁴ : CharZero k
      inst✝³ : Module.Finite k V
      inst✝² : Nontrivial V
      inst✝¹ : LieAlgebra.IsSolvable k L
      inst✝ : LieModule.IsTriangularizable k L V
      imL : LieSubalgebra k (Module.End k V) := (LieModule.toEnd k L V).range
      toEndo : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem imL x) := …
      χ : Module.Dual k (Subtype fun x => Membership.mem imL x)
      h : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑χ) x)
      v : V
      hv✝ : Membership.mem (LieModule.weightSpace V ⇑χ) v
      hv : ∀ (x : Subtype fun x => Membership.mem imL x), Eq (Bracket.bracket x v) ( …
      hv0 : Ne ⟨v, hv✝⟩ 0
      x : L
      ⊢ Eq (Bracket.bracket x v) (HSMul.hSMul ((LinearMap.comp χ toEndo) x) v)
    -/
    apply hv (toEndo x)
    /-
      🎉 no goals
    -/
    /-
      case h.intro.mk.refine_2
      k : Type u_1
      inst✝¹¹ : Field k
      L : Type u_2
      inst✝¹⁰ : LieRing L
      inst✝⁹ : LieAlgebra k L
      V : Type u_3
      inst✝⁸ : AddCommGroup V
      inst✝⁷ : Module k V
      inst✝⁶ : LieRingModule L V
      inst✝⁵ : LieModule k L V
      inst✝⁴ : CharZero k
      inst✝³ : Module.Finite k V
      inst✝² : Nontrivial V
      inst✝¹ : LieAlgebra.IsSolvable k L
      inst✝ : LieModule.IsTriangularizable k L V
      imL : LieSubalgebra k (Module.End k V) := (LieModule.toEnd k L V).range
      toEndo : LinearMap (RingHom.id k) L (Subtype fun x => Membership.mem imL x) := …
      χ : Module.Dual k (Subtype fun x => Membership.mem imL x)
      h : Nontrivial (Subtype fun x => Membership.mem (LieModule.weightSpace V ⇑χ) x)
      v : V
      hv : Membership.mem (LieModule.weightSpace V ⇑χ) v
      hv0 : Ne ⟨v, hv⟩ 0
      ⊢ Ne ⟨v, ⋯⟩ 0
    -/
  · simpa using hv0
    /-
      🎉 no goals
    -/


