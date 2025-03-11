/-- A polynomial `Splits` iff it is zero or all of its irreducible factors have `degree` 1. -/
def Splits (f : K[X]) : Prop :=
  f.map i = 0 ∨ ∀ {g : L[X]}, Irreducible g → g ∣ f.map i → degree g = 1


@[simp]
theorem splits_zero : Splits i (0 : K[X]) :=
  Or.inl (Polynomial.map_zero i)


theorem splits_of_map_eq_C {f : K[X]} {a : L} (h : f.map i = C a) : Splits i f :=
  letI := Classical.decEq L
  if ha : a = 0 then Or.inl (h.trans (ha.symm ▸ C_0))
  else
    Or.inr fun hg ⟨p, hp⟩ =>
      absurd hg.1 <|
        Classical.not_not.2 <|
          isUnit_iff_degree_eq_zero.2 <| by
            /-
              K : Type v
              L : Type w
              inst✝¹ : CommRing K
              inst✝ : Field L
              i : RingHom K L
              f : Polynomial K
              a : L
              h : Eq (Polynomial.map i f) (Polynomial.C a)
              this : DecidableEq L := Classical.decEq L
              ha : Not (Eq a 0)
              g✝ : Polynomial L
              hg : Irreducible g✝
              x✝ : Dvd.dvd g✝ (Polynomial.map i f)
              p : Polynomial L
              hp : Eq (Polynomial.map i f) (HMul.hMul g✝ p)
              ⊢ Eq g✝.degree 0
            -/
            have := congr_arg degree hp
            rw [h, degree_C ha, degree_mul, @eq_comm (WithBot ℕ) 0,
                Nat.WithBot.add_eq_zero_iff] at this
            /-
              K : Type v
              L : Type w
              inst✝¹ : CommRing K
              inst✝ : Field L
              i : RingHom K L
              f : Polynomial K
              a : L
              h : Eq (Polynomial.map i f) (Polynomial.C a)
              this✝ : DecidableEq L := Classical.decEq L
              ha : Not (Eq a 0)
              g✝ : Polynomial L
              hg : Irreducible g✝
              x✝ : Dvd.dvd g✝ (Polynomial.map i f)
              p : Polynomial L
              hp : Eq (Polynomial.map i f) (HMul.hMul g✝ p)
              this : And (Eq g✝.degree 0) (Eq p.degree 0)
              ⊢ Eq g✝.degree 0
            -/
            exact this.1
            /-
              🎉 no goals
            -/


@[simp]
theorem splits_C (a : K) : Splits i (C a) :=
  splits_of_map_eq_C i (map_C i)


theorem splits_of_map_degree_eq_one {f : K[X]} (hf : degree (f.map i) = 1) : Splits i f :=
  Or.inr fun hg ⟨p, hp⟩ => by
    /-
      K : Type v
      L : Type w
      inst✝¹ : CommRing K
      inst✝ : Field L
      i : RingHom K L
      f : Polynomial K
      hf : Eq (Polynomial.map i f).degree 1
      g✝ : Polynomial L
      hg : Irreducible g✝
      x✝ : Dvd.dvd g✝ (Polynomial.map i f)
      p : Polynomial L
      hp : Eq (Polynomial.map i f) (HMul.hMul g✝ p)
      ⊢ Eq g✝.degree 1
    -/
    have := congr_arg degree hp
    simp [Nat.WithBot.add_eq_one_iff, hf, @eq_comm (WithBot ℕ) 1,
        mt isUnit_iff_degree_eq_zero.2 hg.1] at this
    /-
      K : Type v
      L : Type w
      inst✝¹ : CommRing K
      inst✝ : Field L
      i : RingHom K L
      f : Polynomial K
      hf : Eq (Polynomial.map i f).degree 1
      g✝ : Polynomial L
      hg : Irreducible g✝
      x✝ : Dvd.dvd g✝ (Polynomial.map i f)
      p : Polynomial L
      hp : Eq (Polynomial.map i f) (HMul.hMul g✝ p)
      this : And (Eq g✝.degree 1) (Eq p.degree 0)
      ⊢ Eq g✝.degree 1
    -/
    tauto
    /-
      🎉 no goals
    -/


theorem splits_of_degree_le_one {f : K[X]} (hf : degree f ≤ 1) : Splits i f :=
  if hif : degree (f.map i) ≤ 0 then splits_of_map_eq_C i (degree_le_zero_iff.mp hif)
  else by
    /-
      K : Type v
      L : Type w
      inst✝¹ : CommRing K
      inst✝ : Field L
      i : RingHom K L
      f : Polynomial K
      hf : LE.le f.degree 1
      hif : Not (LE.le (Polynomial.map i f).degree 0)
      ⊢ Polynomial.Splits i f
    -/
    push_neg at hif
    /-
      K : Type v
      L : Type w
      inst✝¹ : CommRing K
      inst✝ : Field L
      i : RingHom K L
      f : Polynomial K
      hf : LE.le f.degree 1
      hif : LT.lt 0 (Polynomial.map i f).degree
      ⊢ Polynomial.Splits i f
    -/
    rw [← Order.succ_le_iff, ← WithBot.coe_zero, WithBot.orderSucc_coe, Nat.succ_eq_succ] at hif
    /-
      K : Type v
      L : Type w
      inst✝¹ : CommRing K
      inst✝ : Field L
      i : RingHom K L
      f : Polynomial K
      hf : LE.le f.degree 1
      hif : LE.le (↑(Nat.succ 0)) (Polynomial.map i f).degree
      ⊢ Polynomial.Splits i f
    -/
    exact splits_of_map_degree_eq_one i ((degree_map_le.trans hf).antisymm hif)
    /-
      🎉 no goals
    -/


theorem splits_of_degree_eq_one {f : K[X]} (hf : degree f = 1) : Splits i f :=
  splits_of_degree_le_one i hf.le


theorem splits_of_natDegree_le_one {f : K[X]} (hf : natDegree f ≤ 1) : Splits i f :=
  splits_of_degree_le_one i (degree_le_of_natDegree_le hf)


theorem splits_of_natDegree_eq_one {f : K[X]} (hf : natDegree f = 1) : Splits i f :=
  splits_of_natDegree_le_one i (le_of_eq hf)


theorem splits_mul {f g : K[X]} (hf : Splits i f) (hg : Splits i g) : Splits i (f * g) :=
  letI := Classical.decEq L
  if h : (f * g).map i = 0 then Or.inl h
  else
    Or.inr @fun p hp hpf =>
      ((irreducible_iff_prime.1 hp).2.2 _ _
                                           /-
                                             K : Type v
                                             L : Type w
                                             inst✝¹ : CommRing K
                                             inst✝ : Field L
                                             i : RingHom K L
                                             f g : Polynomial K
                                             hf : Polynomial.Splits i f
                                             hg : Polynomial.Splits i g
                                             this : DecidableEq L := Classical.decEq L
                                             h : Not (Eq (Polynomial.map i (HMul.hMul f g)) 0)
                                             p : Polynomial L
                                             hp : Irreducible p
                                             hpf : Dvd.dvd p (Polynomial.map i (HMul.hMul f g))
                                             ⊢ Dvd.dvd p (HMul.hMul (Polynomial.map i f) (Polynomial.map i g))
                                           -/
            (show p ∣ map i f * map i g by convert hpf; rw [Polynomial.map_mul])).elim
                                                        /-
                                                          🎉 no goals
                                                        -/
                                       /-
                                         K : Type v
                                         L : Type w
                                         inst✝¹ : CommRing K
                                         inst✝ : Field L
                                         i : RingHom K L
                                         f g : Polynomial K
                                         hf✝ : Polynomial.Splits i f
                                         hg : Polynomial.Splits i g
                                         this : DecidableEq L := Classical.decEq L
                                         h : Not (Eq (Polynomial.map i (HMul.hMul f g)) 0)
                                         p : Polynomial L
                                         hp : Irreducible p
                                         hpf : Dvd.dvd p (Polynomial.map i (HMul.hMul f g))
                                         hf : Eq (Polynomial.map i f) 0
                                         ⊢ False
                                       -/
        (hf.resolve_left (fun hf => by simp [hf] at h) hp)
                                       /-
                                         🎉 no goals
                                       -/
                                       /-
                                         K : Type v
                                         L : Type w
                                         inst✝¹ : CommRing K
                                         inst✝ : Field L
                                         i : RingHom K L
                                         f g : Polynomial K
                                         hf : Polynomial.Splits i f
                                         hg✝ : Polynomial.Splits i g
                                         this : DecidableEq L := Classical.decEq L
                                         h : Not (Eq (Polynomial.map i (HMul.hMul f g)) 0)
                                         p : Polynomial L
                                         hp : Irreducible p
                                         hpf : Dvd.dvd p (Polynomial.map i (HMul.hMul f g))
                                         hg : Eq (Polynomial.map i g) 0
                                         ⊢ False
                                       -/
        (hg.resolve_left (fun hg => by simp [hg] at h) hp)
                                       /-
                                         🎉 no goals
                                       -/


theorem splits_of_splits_mul' {f g : K[X]} (hfg : (f * g).map i ≠ 0) (h : Splits i (f * g)) :
    Splits i f ∧ Splits i g :=
  ⟨Or.inr @fun g hgi hg =>
                                    /-
                                      K : Type v
                                      L : Type w
                                      inst✝¹ : CommRing K
                                      inst✝ : Field L
                                      i : RingHom K L
                                      f g✝ : Polynomial K
                                      hfg : Ne (Polynomial.map i (HMul.hMul f g✝)) 0
                                      h : Polynomial.Splits i (HMul.hMul f g✝)
                                      g : Polynomial L
                                      hgi : Irreducible g
                                      hg : Dvd.dvd g (Polynomial.map i f)
                                      ⊢ Dvd.dvd g (Polynomial.map i (HMul.hMul f g✝))
                                    -/
      Or.resolve_left h hfg hgi (by rw [Polynomial.map_mul]; exact hg.trans (dvd_mul_right _ _)),
                                                             /-
                                                               🎉 no goals
                                                             -/
    Or.inr @fun g hgi hg =>
                                    /-
                                      K : Type v
                                      L : Type w
                                      inst✝¹ : CommRing K
                                      inst✝ : Field L
                                      i : RingHom K L
                                      f g✝ : Polynomial K
                                      hfg : Ne (Polynomial.map i (HMul.hMul f g✝)) 0
                                      h : Polynomial.Splits i (HMul.hMul f g✝)
                                      g : Polynomial L
                                      hgi : Irreducible g
                                      hg : Dvd.dvd g (Polynomial.map i g✝)
                                      ⊢ Dvd.dvd g (Polynomial.map i (HMul.hMul f g✝))
                                    -/
      Or.resolve_left h hfg hgi (by rw [Polynomial.map_mul]; exact hg.trans (dvd_mul_left _ _))⟩
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem splits_map_iff (j : L →+* F) {f : K[X]} : Splits j (f.map i) ↔ Splits (j.comp i) f := by
  /-
    F : Type u
    K : Type v
    L : Type w
    inst✝² : CommRing K
    inst✝¹ : Field L
    inst✝ : Field F
    i : RingHom K L
    j : RingHom L F
    f : Polynomial K
    ⊢ Iff (Polynomial.Splits j (Polynomial.map i f)) (Polynomial.Splits (j.comp i) …
  -/
  simp [Splits, Polynomial.map_map]
  /-
    🎉 no goals
  -/


theorem splits_one : Splits i 1 :=
  splits_C i 1


theorem splits_of_isUnit [IsDomain K] {u : K[X]} (hu : IsUnit u) : u.Splits i :=
  (isUnit_iff.mp hu).choose_spec.2 ▸ splits_C _ _


theorem splits_X_sub_C {x : K} : (X - C x).Splits i :=
  splits_of_degree_le_one _ <| degree_X_sub_C_le _


theorem splits_X : X.Splits i :=
  splits_of_degree_le_one _ degree_X_le


theorem splits_prod {ι : Type u} {s : ι → K[X]} {t : Finset ι} :
    (∀ j ∈ t, (s j).Splits i) → (∏ x ∈ t, s x).Splits i := by
  classical
  refine Finset.induction_on t (fun _ => splits_one i) fun a t hat ih ht => ?_
  rw [Finset.forall_mem_insert] at ht; rw [Finset.prod_insert hat]
  exact splits_mul i ht.1 (ih ht.2)


theorem splits_pow {f : K[X]} (hf : f.Splits i) (n : ℕ) : (f ^ n).Splits i := by
  /-
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    i : RingHom K L
    f : Polynomial K
    hf : Polynomial.Splits i f
    n : Nat
    ⊢ Polynomial.Splits i (HPow.hPow f n)
  -/
  rw [← Finset.card_range n, ← Finset.prod_const]
  /-
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    i : RingHom K L
    f : Polynomial K
    hf : Polynomial.Splits i f
    n : Nat
    ⊢ Polynomial.Splits i ((Finset.range n).prod fun _x => f)
  -/
  exact splits_prod i fun j _ => hf
  /-
    🎉 no goals
  -/


theorem splits_X_pow (n : ℕ) : (X ^ n).Splits i :=
  splits_pow i (splits_X i) n


theorem splits_id_iff_splits {f : K[X]} : (f.map i).Splits (RingHom.id L) ↔ f.Splits i := by
  /-
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    i : RingHom K L
    f : Polynomial K
    ⊢ Iff (Polynomial.Splits (RingHom.id L) (Polynomial.map i f)) (Polynomial.Spli …
  -/
  rw [splits_map_iff, RingHom.id_comp]
  /-
    🎉 no goals
  -/


theorem Splits.comp_of_map_degree_le_one {f : K[X]} {p : K[X]} (hd : (p.map i).degree ≤ 1)
    (h : f.Splits i) : (f.comp p).Splits i := by
  /-
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    i : RingHom K L
    f p : Polynomial K
    hd : LE.le (Polynomial.map i p).degree 1
    h : Polynomial.Splits i f
    ⊢ Polynomial.Splits i (f.comp p)
  -/
  by_cases hzero : map i (f.comp p) = 0
    /-
      case pos
      K : Type v
      L : Type w
      inst✝¹ : CommRing K
      inst✝ : Field L
      i : RingHom K L
      f p : Polynomial K
      hd : LE.le (Polynomial.map i p).degree 1
      h : Polynomial.Splits i f
      hzero : Eq (Polynomial.map i (f.comp p)) 0
      ⊢ Polynomial.Splits i (f.comp p)
    -/
  · exact Or.inl hzero
    /-
      🎉 no goals
    -/
  cases h with
  | inl h0 =>
    exact Or.inl <| map_comp i _ _ ▸ h0.symm ▸ zero_comp
  | inr h =>
    right
    intro g irr dvd
    rw [map_comp] at dvd hzero
    cases lt_or_eq_of_le hd with
    | inl hd =>
      rw [eq_C_of_degree_le_zero (Nat.WithBot.lt_one_iff_le_zero.mp hd), comp_C] at dvd hzero
      refine False.elim (irr.1 (isUnit_of_dvd_unit dvd ?_))
      simpa using hzero
    | inr hd =>
      let _ := invertibleOfNonzero (leadingCoeff_ne_zero.mpr
          (ne_zero_of_degree_gt (n := ⊥) (by rw [hd]; decide)))
      rw [eq_X_add_C_of_degree_eq_one hd, dvd_comp_C_mul_X_add_C_iff _ _] at dvd
      have := h (irr.map (algEquivCMulXAddC _ _).symm) dvd
      rw [degree_eq_natDegree irr.ne_zero]
      rwa [algEquivCMulXAddC_symm_apply, ← comp_eq_aeval,
        degree_eq_natDegree (fun h => WithBot.bot_ne_one (h ▸ this)),
        natDegree_comp, natDegree_C_mul (invertibleInvOf.ne_zero),
        natDegree_X_sub_C, mul_one] at this


theorem splits_iff_comp_splits_of_degree_eq_one {f : K[X]} {p : K[X]} (hd : (p.map i).degree = 1) :
    f.Splits i ↔ (f.comp p).Splits i := by
  /-
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    i : RingHom K L
    f p : Polynomial K
    hd : Eq (Polynomial.map i p).degree 1
    ⊢ Iff (Polynomial.Splits i f) (Polynomial.Splits i (f.comp p))
  -/
  rw [← splits_id_iff_splits, ← splits_id_iff_splits (f := f.comp p), map_comp]
  refine ⟨fun h => Splits.comp_of_map_degree_le_one
    (le_of_eq (map_id (R := L) ▸ hd)) h, fun h => ?_⟩
  let _ := invertibleOfNonzero (leadingCoeff_ne_zero.mpr
      (ne_zero_of_degree_gt (n := ⊥) (by rw [hd]; decide)))
  have : (map i f) = ((map i f).comp (map i p)).comp ((C ⅟ (map i p).leadingCoeff *
      (X - C ((map i p).coeff 0)))) := by
    rw [comp_assoc]
    nth_rw 1 [eq_X_add_C_of_degree_eq_one hd]
    simp only [coeff_map, invOf_eq_inv, mul_sub, ← C_mul, add_comp, mul_comp, C_comp, X_comp,
      ← mul_assoc]
    simp
  /-
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    i : RingHom K L
    f p : Polynomial K
    hd : Eq (Polynomial.map i p).degree 1
    h : Polynomial.Splits (RingHom.id L) ((Polynomial.map i f).comp (Polynomial.ma …
    x✝ : Invertible (Polynomial.map i p).leadingCoeff := invertibleOfNonzero ⋯
    this : Eq (Polynomial.map i f) (((Polynomial.map i f).comp (Polynomial.map i p …
    ⊢ Polynomial.Splits (RingHom.id L) (Polynomial.map i f)
  -/
  refine this ▸ Splits.comp_of_map_degree_le_one ?_ h
  /-
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    i : RingHom K L
    f p : Polynomial K
    hd : Eq (Polynomial.map i p).degree 1
    h : Polynomial.Splits (RingHom.id L) ((Polynomial.map i f).comp (Polynomial.ma …
    x✝ : Invertible (Polynomial.map i p).leadingCoeff := invertibleOfNonzero ⋯
    this : Eq (Polynomial.map i f) (((Polynomial.map i f).comp (Polynomial.map i p …
    ⊢ LE.le (Polynomial.map (RingHom.id L) (HMul.hMul (Polynomial.C (Invertible.in …
  -/
  simp [degree_C (inv_ne_zero (Invertible.ne_zero (a := (map i p).leadingCoeff)))]
  /-
    🎉 no goals
  -/


/--
This is a weaker variant of `Splits.comp_of_map_degree_le_one`,
but its conditions are easier to check.
-/
theorem Splits.comp_of_degree_le_one {f : K[X]} {p : K[X]} (hd : p.degree ≤ 1)
    (h : f.Splits i) : (f.comp p).Splits i :=
  Splits.comp_of_map_degree_le_one (degree_map_le.trans hd) h


theorem Splits.comp_X_sub_C (a : K) {f : K[X]}
    (h : f.Splits i) : (f.comp (X - C a)).Splits i :=
  Splits.comp_of_degree_le_one (degree_X_sub_C_le _) h


theorem Splits.comp_X_add_C (a : K) {f : K[X]}
    (h : f.Splits i) : (f.comp (X + C a)).Splits i :=
                                   /-
                                     K : Type v
                                     L : Type w
                                     inst✝¹ : CommRing K
                                     inst✝ : Field L
                                     i : RingHom K L
                                     a : K
                                     f : Polynomial K
                                     h : Polynomial.Splits i f
                                     ⊢ LE.le (HAdd.hAdd Polynomial.X (Polynomial.C a)).degree 1
                                   -/
  Splits.comp_of_degree_le_one (by simpa using degree_X_sub_C_le (-a)) h
                                   /-
                                     🎉 no goals
                                   -/


theorem Splits.comp_neg_X {f : K[X]} (h : f.Splits i) : (f.comp (-X)).Splits i :=
                                   /-
                                     K : Type v
                                     L : Type w
                                     inst✝¹ : CommRing K
                                     inst✝ : Field L
                                     i : RingHom K L
                                     f : Polynomial K
                                     h : Polynomial.Splits i f
                                     ⊢ LE.le (Neg.neg Polynomial.X).degree 1
                                   -/
  Splits.comp_of_degree_le_one (by simpa using degree_X_sub_C_le (0 : K)) h
                                   /-
                                     🎉 no goals
                                   -/


theorem exists_root_of_splits' {f : K[X]} (hs : Splits i f) (hf0 : degree (f.map i) ≠ 0) :
    ∃ x, eval₂ i x f = 0 :=
  letI := Classical.decEq L
                                /-
                                  K : Type v
                                  L : Type w
                                  inst✝¹ : CommRing K
                                  inst✝ : Field L
                                  i : RingHom K L
                                  f : Polynomial K
                                  hs : Polynomial.Splits i f
                                  hf0 : Ne (Polynomial.map i f).degree 0
                                  this : DecidableEq L := Classical.decEq L
                                  hf0' : Eq (Polynomial.map i f) 0
                                  ⊢ Exists fun x => Eq (Polynomial.eval₂ i x f) 0
                                -/
  if hf0' : f.map i = 0 then by simp [eval₂_eq_eval_map, hf0']
                                /-
                                  🎉 no goals
                                -/
  else
    let ⟨g, hg⟩ :=
      WfDvdMonoid.exists_irreducible_factor
        (show ¬IsUnit (f.map i) from mt isUnit_iff_degree_eq_zero.1 hf0) hf0'
    let ⟨x, hx⟩ := exists_root_of_degree_eq_one (hs.resolve_left hf0' hg.1 hg.2)
    let ⟨i, hi⟩ := hg.2
           /-
             K : Type v
             L : Type w
             inst✝¹ : CommRing K
             inst✝ : Field L
             i✝ : RingHom K L
             f : Polynomial K
             hs : Polynomial.Splits i✝ f
             hf0 : Ne (Polynomial.map i✝ f).degree 0
             this : DecidableEq L := Classical.decEq L
             hf0' : Not (Eq (Polynomial.map i✝ f) 0)
             g : Polynomial L
             hg : And (Irreducible g) (Dvd.dvd g (Polynomial.map i✝ f))
             x : L
             hx : g.IsRoot x
             i : Polynomial L
             hi : Eq (Polynomial.map i✝ f) (HMul.hMul g i)
             ⊢ Eq (Polynomial.eval₂ i✝ x f) 0
           -/
    ⟨x, by rw [← eval_map, hi, eval_mul, show _ = _ from hx, zero_mul]⟩
           /-
             🎉 no goals
           -/


theorem roots_ne_zero_of_splits' {f : K[X]} (hs : Splits i f) (hf0 : natDegree (f.map i) ≠ 0) :
    (f.map i).roots ≠ 0 :=
  let ⟨x, hx⟩ := exists_root_of_splits' i hs fun h => hf0 <| natDegree_eq_of_degree_eq_some h
  fun h => by
  /-
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    i : RingHom K L
    f : Polynomial K
    hs : Polynomial.Splits i f
    hf0 : Ne (Polynomial.map i f).natDegree 0
    x : L
    hx : Eq (Polynomial.eval₂ i x f) 0
    h : Eq (Polynomial.map i f).roots 0
    ⊢ False
  -/
  rw [← eval_map] at hx
  /-
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    i : RingHom K L
    f : Polynomial K
    hs : Polynomial.Splits i f
    hf0 : Ne (Polynomial.map i f).natDegree 0
    x : L
    hx : Eq (Polynomial.eval x (Polynomial.map i f)) 0
    h : Eq (Polynomial.map i f).roots 0
    ⊢ False
  -/
  have : f.map i ≠ 0 := by intro; simp_all
  /-
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    i : RingHom K L
    f : Polynomial K
    hs : Polynomial.Splits i f
    hf0 : Ne (Polynomial.map i f).natDegree 0
    x : L
    hx : Eq (Polynomial.eval x (Polynomial.map i f)) 0
    h : Eq (Polynomial.map i f).roots 0
    this : Ne (Polynomial.map i f) 0
    ⊢ False
  -/
  cases h.subst ((mem_roots this).2 hx)
  /-
    🎉 no goals
  -/


/-- Pick a root of a polynomial that splits. See `rootOfSplits` for polynomials over a field
which has simpler assumptions. -/
def rootOfSplits' {f : K[X]} (hf : f.Splits i) (hfd : (f.map i).degree ≠ 0) : L :=
  Classical.choose <| exists_root_of_splits' i hf hfd


theorem map_rootOfSplits' {f : K[X]} (hf : f.Splits i) (hfd) :
    f.eval₂ i (rootOfSplits' i hf hfd) = 0 :=
  Classical.choose_spec <| exists_root_of_splits' i hf hfd


theorem natDegree_eq_card_roots' {p : K[X]} {i : K →+* L} (hsplit : Splits i p) :
    (p.map i).natDegree = Multiset.card (p.map i).roots := by
  /-
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    p : Polynomial K
    i : RingHom K L
    hsplit : Polynomial.Splits i p
    ⊢ Eq (Polynomial.map i p).natDegree (Polynomial.map i p).roots.card
  -/
  by_cases hp : p.map i = 0
    /-
      case pos
      K : Type v
      L : Type w
      inst✝¹ : CommRing K
      inst✝ : Field L
      p : Polynomial K
      i : RingHom K L
      hsplit : Polynomial.Splits i p
      hp : Eq (Polynomial.map i p) 0
      ⊢ Eq (Polynomial.map i p).natDegree (Polynomial.map i p).roots.card
    -/
  · rw [hp, natDegree_zero, roots_zero, Multiset.card_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    p : Polynomial K
    i : RingHom K L
    hsplit : Polynomial.Splits i p
    hp : Not (Eq (Polynomial.map i p) 0)
    ⊢ Eq (Polynomial.map i p).natDegree (Polynomial.map i p).roots.card
  -/
  obtain ⟨q, he, hd, hr⟩ := exists_prod_multiset_X_sub_C_mul (p.map i)
  /-
    case neg.intro.intro.intro
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    p : Polynomial K
    i : RingHom K L
    hsplit : Polynomial.Splits i p
    hp : Not (Eq (Polynomial.map i p) 0)
    q : Polynomial L
    he : Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial. …
    hd : Eq (HAdd.hAdd (Polynomial.map i p).roots.card q.natDegree) (Polynomial.ma …
    hr : Eq q.roots 0
    ⊢ Eq (Polynomial.map i p).natDegree (Polynomial.map i p).roots.card
  -/
  rw [← splits_id_iff_splits, ← he] at hsplit
  /-
    case neg.intro.intro.intro
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    p : Polynomial K
    i : RingHom K L
    hp : Not (Eq (Polynomial.map i p) 0)
    q : Polynomial L
    hsplit : Polynomial.Splits (RingHom.id L) (HMul.hMul (Multiset.map (fun a => H …
    he : Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial. …
    hd : Eq (HAdd.hAdd (Polynomial.map i p).roots.card q.natDegree) (Polynomial.ma …
    hr : Eq q.roots 0
    ⊢ Eq (Polynomial.map i p).natDegree (Polynomial.map i p).roots.card
  -/
  rw [← he] at hp
  /-
    case neg.intro.intro.intro
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    p : Polynomial K
    i : RingHom K L
    q : Polynomial L
    hp : Not (Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polyno …
    hsplit : Polynomial.Splits (RingHom.id L) (HMul.hMul (Multiset.map (fun a => H …
    he : Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial. …
    hd : Eq (HAdd.hAdd (Polynomial.map i p).roots.card q.natDegree) (Polynomial.ma …
    hr : Eq q.roots 0
    ⊢ Eq (Polynomial.map i p).natDegree (Polynomial.map i p).roots.card
  -/
  have hq : q ≠ 0 := fun h => hp (by rw [h, mul_zero])
  /-
    case neg.intro.intro.intro
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    p : Polynomial K
    i : RingHom K L
    q : Polynomial L
    hp : Not (Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polyno …
    hsplit : Polynomial.Splits (RingHom.id L) (HMul.hMul (Multiset.map (fun a => H …
    he : Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial. …
    hd : Eq (HAdd.hAdd (Polynomial.map i p).roots.card q.natDegree) (Polynomial.ma …
    hr : Eq q.roots 0
    hq : Ne q 0
    ⊢ Eq (Polynomial.map i p).natDegree (Polynomial.map i p).roots.card
  -/
  rw [← hd, add_right_eq_self]
  /-
    case neg.intro.intro.intro
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    p : Polynomial K
    i : RingHom K L
    q : Polynomial L
    hp : Not (Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polyno …
    hsplit : Polynomial.Splits (RingHom.id L) (HMul.hMul (Multiset.map (fun a => H …
    he : Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial. …
    hd : Eq (HAdd.hAdd (Polynomial.map i p).roots.card q.natDegree) (Polynomial.ma …
    hr : Eq q.roots 0
    hq : Ne q 0
    ⊢ Eq q.natDegree 0
  -/
  by_contra h
  /-
    case neg.intro.intro.intro
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    p : Polynomial K
    i : RingHom K L
    q : Polynomial L
    hp : Not (Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polyno …
    hsplit : Polynomial.Splits (RingHom.id L) (HMul.hMul (Multiset.map (fun a => H …
    he : Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial. …
    hd : Eq (HAdd.hAdd (Polynomial.map i p).roots.card q.natDegree) (Polynomial.ma …
    hr : Eq q.roots 0
    hq : Ne q 0
    h : Not (Eq q.natDegree 0)
    ⊢ False
  -/
  have h' : (map (RingHom.id L) q).natDegree ≠ 0 := by simp [h]
  /-
    case neg.intro.intro.intro
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    p : Polynomial K
    i : RingHom K L
    q : Polynomial L
    hp : Not (Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polyno …
    hsplit : Polynomial.Splits (RingHom.id L) (HMul.hMul (Multiset.map (fun a => H …
    he : Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial. …
    hd : Eq (HAdd.hAdd (Polynomial.map i p).roots.card q.natDegree) (Polynomial.ma …
    hr : Eq q.roots 0
    hq : Ne q 0
    h : Not (Eq q.natDegree 0)
    h' : Ne (Polynomial.map (RingHom.id L) q).natDegree 0
    ⊢ False
  -/
  have := roots_ne_zero_of_splits' (RingHom.id L) (splits_of_splits_mul' _ ?_ hsplit).2 h'
    /-
      case neg.intro.intro.intro.refine_2
      K : Type v
      L : Type w
      inst✝¹ : CommRing K
      inst✝ : Field L
      p : Polynomial K
      i : RingHom K L
      q : Polynomial L
      hp : Not (Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polyno …
      hsplit : Polynomial.Splits (RingHom.id L) (HMul.hMul (Multiset.map (fun a => H …
      he : Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial. …
      hd : Eq (HAdd.hAdd (Polynomial.map i p).roots.card q.natDegree) (Polynomial.ma …
      hr : Eq q.roots 0
      hq : Ne q 0
      h : Not (Eq q.natDegree 0)
      h' : Ne (Polynomial.map (RingHom.id L) q).natDegree 0
      this : Ne (Polynomial.map (RingHom.id L) q).roots 0
      ⊢ False
    -/
  · rw [map_id] at this
    /-
      case neg.intro.intro.intro.refine_2
      K : Type v
      L : Type w
      inst✝¹ : CommRing K
      inst✝ : Field L
      p : Polynomial K
      i : RingHom K L
      q : Polynomial L
      hp : Not (Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polyno …
      hsplit : Polynomial.Splits (RingHom.id L) (HMul.hMul (Multiset.map (fun a => H …
      he : Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial. …
      hd : Eq (HAdd.hAdd (Polynomial.map i p).roots.card q.natDegree) (Polynomial.ma …
      hr : Eq q.roots 0
      hq : Ne q 0
      h : Not (Eq q.natDegree 0)
      h' : Ne (Polynomial.map (RingHom.id L) q).natDegree 0
      this : Ne q.roots 0
      ⊢ False
    -/
    exact this hr
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.intro.intro.refine_1
      K : Type v
      L : Type w
      inst✝¹ : CommRing K
      inst✝ : Field L
      p : Polynomial K
      i : RingHom K L
      q : Polynomial L
      hp : Not (Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polyno …
      hsplit : Polynomial.Splits (RingHom.id L) (HMul.hMul (Multiset.map (fun a => H …
      he : Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial. …
      hd : Eq (HAdd.hAdd (Polynomial.map i p).roots.card q.natDegree) (Polynomial.ma …
      hr : Eq q.roots 0
      hq : Ne q 0
      h : Not (Eq q.natDegree 0)
      h' : Ne (Polynomial.map (RingHom.id L) q).natDegree 0
      ⊢ Ne (Polynomial.map (RingHom.id L) (HMul.hMul (Multiset.map (fun a => HSub.hS …
    -/
  · rw [map_id]
    /-
      case neg.intro.intro.intro.refine_1
      K : Type v
      L : Type w
      inst✝¹ : CommRing K
      inst✝ : Field L
      p : Polynomial K
      i : RingHom K L
      q : Polynomial L
      hp : Not (Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polyno …
      hsplit : Polynomial.Splits (RingHom.id L) (HMul.hMul (Multiset.map (fun a => H …
      he : Eq (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial. …
      hd : Eq (HAdd.hAdd (Polynomial.map i p).roots.card q.natDegree) (Polynomial.ma …
      hr : Eq q.roots 0
      hq : Ne q 0
      h : Not (Eq q.natDegree 0)
      h' : Ne (Polynomial.map (RingHom.id L) q).natDegree 0
      ⊢ Ne (HMul.hMul (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a …
    -/
    exact mul_ne_zero monic_prod_multiset_X_sub_C.ne_zero hq
    /-
      🎉 no goals
    -/


theorem degree_eq_card_roots' {p : K[X]} {i : K →+* L} (p_ne_zero : p.map i ≠ 0)
    (hsplit : Splits i p) : (p.map i).degree = Multiset.card (p.map i).roots := by
  /-
    K : Type v
    L : Type w
    inst✝¹ : CommRing K
    inst✝ : Field L
    p : Polynomial K
    i : RingHom K L
    p_ne_zero : Ne (Polynomial.map i p) 0
    hsplit : Polynomial.Splits i p
    ⊢ Eq (Polynomial.map i p).degree ↑(Polynomial.map i p).roots.card
  -/
  simp [degree_eq_natDegree p_ne_zero, natDegree_eq_card_roots' hsplit]
  /-
    🎉 no goals
  -/


/-- This lemma is for polynomials over a field. -/
theorem splits_iff (f : K[X]) :
    Splits i f ↔ f = 0 ∨ ∀ {g : L[X]}, Irreducible g → g ∣ f.map i → degree g = 1 := by
  /-
    K : Type v
    L : Type w
    inst✝¹ : Field K
    inst✝ : Field L
    i : RingHom K L
    f : Polynomial K
    ⊢ Iff (Polynomial.Splits i f) (Or (Eq f 0) (∀ {g : Polynomial L}, Irreducible  …
  -/
  rw [Splits, Polynomial.map_eq_zero]
  /-
    🎉 no goals
  -/


/-- This lemma is for polynomials over a field. -/
theorem Splits.def {i : K →+* L} {f : K[X]} (h : Splits i f) :
    f = 0 ∨ ∀ {g : L[X]}, Irreducible g → g ∣ f.map i → degree g = 1 :=
  (splits_iff i f).mp h


theorem splits_of_splits_mul {f g : K[X]} (hfg : f * g ≠ 0) (h : Splits i (f * g)) :
    Splits i f ∧ Splits i g :=
  splits_of_splits_mul' i (map_ne_zero hfg) h


theorem splits_of_splits_of_dvd {f g : K[X]} (hf0 : f ≠ 0) (hf : Splits i f) (hgf : g ∣ f) :
    Splits i g := by
  /-
    K : Type v
    L : Type w
    inst✝¹ : Field K
    inst✝ : Field L
    i : RingHom K L
    f g : Polynomial K
    hf0 : Ne f 0
    hf : Polynomial.Splits i f
    hgf : Dvd.dvd g f
    ⊢ Polynomial.Splits i g
  -/
  obtain ⟨f, rfl⟩ := hgf
  /-
    case intro
    K : Type v
    L : Type w
    inst✝¹ : Field K
    inst✝ : Field L
    i : RingHom K L
    g f : Polynomial K
    hf0 : Ne (HMul.hMul g f) 0
    hf : Polynomial.Splits i (HMul.hMul g f)
    ⊢ Polynomial.Splits i g
  -/
  exact (splits_of_splits_mul i hf0 hf).1
  /-
    🎉 no goals
  -/


theorem splits_of_splits_gcd_left [DecidableEq K] {f g : K[X]} (hf0 : f ≠ 0) (hf : Splits i f) :
    Splits i (EuclideanDomain.gcd f g) :=
  Polynomial.splits_of_splits_of_dvd i hf0 hf (EuclideanDomain.gcd_dvd_left f g)


theorem splits_of_splits_gcd_right [DecidableEq K] {f g : K[X]} (hg0 : g ≠ 0) (hg : Splits i g) :
    Splits i (EuclideanDomain.gcd f g) :=
  Polynomial.splits_of_splits_of_dvd i hg0 hg (EuclideanDomain.gcd_dvd_right f g)


theorem splits_mul_iff {f g : K[X]} (hf : f ≠ 0) (hg : g ≠ 0) :
    (f * g).Splits i ↔ f.Splits i ∧ g.Splits i :=
  ⟨splits_of_splits_mul i (mul_ne_zero hf hg), fun ⟨hfs, hgs⟩ => splits_mul i hfs hgs⟩


theorem splits_prod_iff {ι : Type u} {s : ι → K[X]} {t : Finset ι} :
    (∀ j ∈ t, s j ≠ 0) → ((∏ x ∈ t, s x).Splits i ↔ ∀ j ∈ t, (s j).Splits i) := by
  classical
  refine
    Finset.induction_on t (fun _ =>
        ⟨fun _ _ h => by simp only [Finset.not_mem_empty] at h, fun _ => splits_one i⟩)
      fun a t hat ih ht => ?_
  rw [Finset.forall_mem_insert] at ht ⊢
  rw [Finset.prod_insert hat, splits_mul_iff i ht.1 (Finset.prod_ne_zero_iff.2 ht.2), ih ht.2]


theorem degree_eq_one_of_irreducible_of_splits {p : K[X]} (hp : Irreducible p)
    (hp_splits : Splits (RingHom.id K) p) : p.degree = 1 := by
  /-
    K : Type v
    inst✝ : Field K
    p : Polynomial K
    hp : Irreducible p
    hp_splits : Polynomial.Splits (RingHom.id K) p
    ⊢ Eq p.degree 1
  -/
  rcases hp_splits with ⟨⟩ | hp_splits
    /-
      case inl
      K : Type v
      inst✝ : Field K
      p : Polynomial K
      hp : Irreducible p
      h✝ : Eq (Polynomial.map (RingHom.id K) p) 0
      ⊢ Eq p.degree 1
    -/
  · exfalso
    /-
      case inl
      K : Type v
      inst✝ : Field K
      p : Polynomial K
      hp : Irreducible p
      h✝ : Eq (Polynomial.map (RingHom.id K) p) 0
      ⊢ False
    -/
    simp_all
    /-
      🎉 no goals
    -/
    /-
      case inr
      K : Type v
      inst✝ : Field K
      p : Polynomial K
      hp : Irreducible p
      hp_splits : ∀ {g : Polynomial K}, Irreducible g → Dvd.dvd g (Polynomial.map (R …
      ⊢ Eq p.degree 1
    -/
  · apply hp_splits hp
    /-
      case inr
      K : Type v
      inst✝ : Field K
      p : Polynomial K
      hp : Irreducible p
      hp_splits : ∀ {g : Polynomial K}, Irreducible g → Dvd.dvd g (Polynomial.map (R …
      ⊢ Dvd.dvd p (Polynomial.map (RingHom.id K) p)
    -/
    simp
    /-
      🎉 no goals
    -/


theorem exists_root_of_splits {f : K[X]} (hs : Splits i f) (hf0 : degree f ≠ 0) :
    ∃ x, eval₂ i x f = 0 :=
  exists_root_of_splits' i hs ((f.degree_map i).symm ▸ hf0)


theorem roots_ne_zero_of_splits {f : K[X]} (hs : Splits i f) (hf0 : natDegree f ≠ 0) :
    (f.map i).roots ≠ 0 :=
  roots_ne_zero_of_splits' i hs (ne_of_eq_of_ne (natDegree_map i) hf0)


/-- Pick a root of a polynomial that splits. This version is for polynomials over a field and has
simpler assumptions. -/
def rootOfSplits {f : K[X]} (hf : f.Splits i) (hfd : f.degree ≠ 0) : L :=
  rootOfSplits' i hf ((f.degree_map i).symm ▸ hfd)


/-- `rootOfSplits'` is definitionally equal to `rootOfSplits`. -/
theorem rootOfSplits'_eq_rootOfSplits {f : K[X]} (hf : f.Splits i) (hfd) :
    rootOfSplits' i hf hfd = rootOfSplits i hf (f.degree_map i ▸ hfd) :=
  rfl


theorem map_rootOfSplits {f : K[X]} (hf : f.Splits i) (hfd) :
    f.eval₂ i (rootOfSplits i hf hfd) = 0 :=
  map_rootOfSplits' i hf (ne_of_eq_of_ne (degree_map f i) hfd)


theorem natDegree_eq_card_roots {p : K[X]} {i : K →+* L} (hsplit : Splits i p) :
    p.natDegree = Multiset.card (p.map i).roots :=
  (natDegree_map i).symm.trans <| natDegree_eq_card_roots' hsplit


theorem degree_eq_card_roots {p : K[X]} {i : K →+* L} (p_ne_zero : p ≠ 0) (hsplit : Splits i p) :
    p.degree = Multiset.card (p.map i).roots := by
  /-
    K : Type v
    L : Type w
    inst✝¹ : Field K
    inst✝ : Field L
    p : Polynomial K
    i : RingHom K L
    p_ne_zero : Ne p 0
    hsplit : Polynomial.Splits i p
    ⊢ Eq p.degree ↑(Polynomial.map i p).roots.card
  -/
  rw [degree_eq_natDegree p_ne_zero, natDegree_eq_card_roots hsplit]
  /-
    🎉 no goals
  -/


theorem roots_map {f : K[X]} (hf : f.Splits <| RingHom.id K) : (f.map i).roots = f.roots.map i :=
  (roots_map_of_injective_of_card_eq_natDegree i.injective <| by
      /-
        K : Type v
        L : Type w
        inst✝¹ : Field K
        inst✝ : Field L
        i : RingHom K L
        f : Polynomial K
        hf : Polynomial.Splits (RingHom.id K) f
        ⊢ Eq f.roots.card f.natDegree
      -/
      convert (natDegree_eq_card_roots hf).symm
      /-
        case h.e'_2.h.e'_2.h.e'_4
        K : Type v
        L : Type w
        inst✝¹ : Field K
        inst✝ : Field L
        i : RingHom K L
        f : Polynomial K
        hf : Polynomial.Splits (RingHom.id K) f
        ⊢ Eq f (Polynomial.map (RingHom.id K) f)
      -/
      rw [map_id]).symm
      /-
        🎉 no goals
      -/


theorem image_rootSet [Algebra R K] [Algebra R L] {p : R[X]} (h : p.Splits (algebraMap R K))
    (f : K →ₐ[R] L) : f '' p.rootSet K = p.rootSet L := by
  classical
    rw [rootSet, ← Finset.coe_image, ← Multiset.toFinset_map, ← f.coe_toRingHom,
      ← roots_map _ ((splits_id_iff_splits (algebraMap R K)).mpr h), map_map, f.comp_algebraMap,
      ← rootSet]


theorem adjoin_rootSet_eq_range [Algebra R K] [Algebra R L] {p : R[X]}
    (h : p.Splits (algebraMap R K)) (f : K →ₐ[R] L) :
    Algebra.adjoin R (p.rootSet L) = f.range ↔ Algebra.adjoin R (p.rootSet K) = ⊤ := by
  /-
    R : Type u_1
    K : Type v
    L : Type w
    inst✝⁴ : CommRing R
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra R K
    inst✝ : Algebra R L
    p : Polynomial R
    h : Polynomial.Splits (algebraMap R K) p
    f : AlgHom R K L
    ⊢ Iff (Eq (Algebra.adjoin R (p.rootSet L)) f.range) (Eq (Algebra.adjoin R (p.r …
  -/
  rw [← image_rootSet h f, Algebra.adjoin_image, ← Algebra.map_top]
  /-
    R : Type u_1
    K : Type v
    L : Type w
    inst✝⁴ : CommRing R
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra R K
    inst✝ : Algebra R L
    p : Polynomial R
    h : Polynomial.Splits (algebraMap R K) p
    f : AlgHom R K L
    ⊢ Iff (Eq (Subalgebra.map f (Algebra.adjoin R (p.rootSet K))) (Subalgebra.map  …
  -/
  exact (Subalgebra.map_injective f.toRingHom.injective).eq_iff
  /-
    🎉 no goals
  -/


theorem eq_prod_roots_of_splits {p : K[X]} {i : K →+* L} (hsplit : Splits i p) :
    p.map i = C (i p.leadingCoeff) * ((p.map i).roots.map fun a => X - C a).prod := by
  /-
    K : Type v
    L : Type w
    inst✝¹ : Field K
    inst✝ : Field L
    p : Polynomial K
    i : RingHom K L
    hsplit : Polynomial.Splits i p
    ⊢ Eq (Polynomial.map i p) (HMul.hMul (Polynomial.C (i p.leadingCoeff)) (Multis …
  -/
  rw [← leadingCoeff_map]; symm
  /-
    K : Type v
    L : Type w
    inst✝¹ : Field K
    inst✝ : Field L
    p : Polynomial K
    i : RingHom K L
    hsplit : Polynomial.Splits i p
    ⊢ Eq (HMul.hMul (Polynomial.C (Polynomial.map i p).leadingCoeff) (Multiset.map …
  -/
  apply C_leadingCoeff_mul_prod_multiset_X_sub_C
  /-
    case hroots
    K : Type v
    L : Type w
    inst✝¹ : Field K
    inst✝ : Field L
    p : Polynomial K
    i : RingHom K L
    hsplit : Polynomial.Splits i p
    ⊢ Eq (Polynomial.map i p).roots.card (Polynomial.map i p).natDegree
  -/
  rw [natDegree_map]; exact (natDegree_eq_card_roots hsplit).symm
                      /-
                        🎉 no goals
                      -/


theorem eq_prod_roots_of_splits_id {p : K[X]} (hsplit : Splits (RingHom.id K) p) :
    p = C p.leadingCoeff * (p.roots.map fun a => X - C a).prod := by
  /-
    K : Type v
    inst✝ : Field K
    p : Polynomial K
    hsplit : Polynomial.Splits (RingHom.id K) p
    ⊢ Eq p (HMul.hMul (Polynomial.C p.leadingCoeff) (Multiset.map (fun a => HSub.h …
  -/
  simpa using eq_prod_roots_of_splits hsplit
  /-
    🎉 no goals
  -/


theorem Splits.dvd_of_roots_le_roots {p q : K[X]} (hp : p.Splits (RingHom.id _)) (hp0 : p ≠ 0)
    (hq : p.roots ≤ q.roots) : p ∣ q := by
  /-
    K : Type v
    inst✝ : Field K
    p q : Polynomial K
    hp : Polynomial.Splits (RingHom.id K) p
    hp0 : Ne p 0
    hq : LE.le p.roots q.roots
    ⊢ Dvd.dvd p q
  -/
  rw [eq_prod_roots_of_splits_id hp, C_mul_dvd (leadingCoeff_ne_zero.2 hp0)]
  exact dvd_trans
    (Multiset.prod_dvd_prod_of_le (Multiset.map_le_map hq))
    (prod_multiset_X_sub_C_dvd _)


theorem Splits.dvd_iff_roots_le_roots {p q : K[X]}
    (hp : p.Splits (RingHom.id _)) (hp0 : p ≠ 0) (hq0 : q ≠ 0) :
    p ∣ q ↔ p.roots ≤ q.roots :=
  ⟨Polynomial.roots.le_of_dvd hq0, hp.dvd_of_roots_le_roots hp0⟩


theorem aeval_eq_prod_aroots_sub_of_splits [Algebra K L] {p : K[X]}
    (hsplit : Splits (algebraMap K L) p) (v : L) :
    aeval v p = algebraMap K L p.leadingCoeff * ((p.aroots L).map fun a ↦ v - a).prod := by
  /-
    K : Type v
    L : Type w
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    p : Polynomial K
    hsplit : Polynomial.Splits (algebraMap K L) p
    v : L
    ⊢ Eq ((Polynomial.aeval v) p) (HMul.hMul ((algebraMap K L) p.leadingCoeff) (Mu …
  -/
  rw [← eval_map_algebraMap, eq_prod_roots_of_splits hsplit]
  /-
    K : Type v
    L : Type w
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    p : Polynomial K
    hsplit : Polynomial.Splits (algebraMap K L) p
    v : L
    ⊢ Eq (Polynomial.eval v (HMul.hMul (Polynomial.C ((algebraMap K L) p.leadingCo …
  -/
  simp [eval_multiset_prod]
  /-
    🎉 no goals
  -/


theorem eval_eq_prod_roots_sub_of_splits_id {p : K[X]}
    (hsplit : Splits (RingHom.id K) p) (v : K) :
    eval v p = p.leadingCoeff * (p.roots.map fun a ↦ v - a).prod := by
  /-
    K : Type v
    inst✝ : Field K
    p : Polynomial K
    hsplit : Polynomial.Splits (RingHom.id K) p
    v : K
    ⊢ Eq (Polynomial.eval v p) (HMul.hMul p.leadingCoeff (Multiset.map (fun a => H …
  -/
  convert aeval_eq_prod_aroots_sub_of_splits hsplit v
  /-
    case h.e'_3.h.e'_6.h.e'_3.a.h.e'_4.h
    K : Type v
    inst✝ : Field K
    p : Polynomial K
    hsplit : Polynomial.Splits (RingHom.id K) p
    v : K
    ⊢ Eq p (Polynomial.map (algebraMap K K) p)
  -/
  rw [Algebra.id.map_eq_id, map_id]
  /-
    🎉 no goals
  -/


theorem eq_prod_roots_of_monic_of_splits_id {p : K[X]} (m : Monic p)
    (hsplit : Splits (RingHom.id K) p) : p = (p.roots.map fun a => X - C a).prod := by
  /-
    K : Type v
    inst✝ : Field K
    p : Polynomial K
    m : p.Monic
    hsplit : Polynomial.Splits (RingHom.id K) p
    ⊢ Eq p (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) p.root …
  -/
  convert eq_prod_roots_of_splits_id hsplit
  /-
    case h.e'_3
    K : Type v
    inst✝ : Field K
    p : Polynomial K
    m : p.Monic
    hsplit : Polynomial.Splits (RingHom.id K) p
    ⊢ Eq (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) p.roots) …
  -/
  simp [m]
  /-
    🎉 no goals
  -/


theorem aeval_eq_prod_aroots_sub_of_monic_of_splits [Algebra K L] {p : K[X]} (m : Monic p)
    (hsplit : Splits (algebraMap K L) p) (v : L) :
    aeval v p = ((p.aroots L).map fun a ↦ v - a).prod := by
  /-
    K : Type v
    L : Type w
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    p : Polynomial K
    m : p.Monic
    hsplit : Polynomial.Splits (algebraMap K L) p
    v : L
    ⊢ Eq ((Polynomial.aeval v) p) (Multiset.map (fun a => HSub.hSub v a) (p.aroots …
  -/
  simp [aeval_eq_prod_aroots_sub_of_splits hsplit, m]
  /-
    🎉 no goals
  -/


theorem eval_eq_prod_roots_sub_of_monic_of_splits_id {p : K[X]} (m : Monic p)
    (hsplit : Splits (RingHom.id K) p) (v : K) :
    eval v p = (p.roots.map fun a ↦ v - a).prod := by
  /-
    K : Type v
    inst✝ : Field K
    p : Polynomial K
    m : p.Monic
    hsplit : Polynomial.Splits (RingHom.id K) p
    v : K
    ⊢ Eq (Polynomial.eval v p) (Multiset.map (fun a => HSub.hSub v a) p.roots).prod
  -/
  simp [eval_eq_prod_roots_sub_of_splits_id hsplit, m]
  /-
    🎉 no goals
  -/


theorem eq_X_sub_C_of_splits_of_single_root {x : K} {h : K[X]} (h_splits : Splits i h)
    (h_roots : (h.map i).roots = {i x}) : h = C h.leadingCoeff * (X - C x) := by
  /-
    K : Type v
    L : Type w
    inst✝¹ : Field K
    inst✝ : Field L
    i : RingHom K L
    x : K
    h : Polynomial K
    h_splits : Polynomial.Splits i h
    h_roots : Eq (Polynomial.map i h).roots (Singleton.singleton (i x))
    ⊢ Eq h (HMul.hMul (Polynomial.C h.leadingCoeff) (HSub.hSub Polynomial.X (Polyn …
  -/
  apply Polynomial.map_injective _ i.injective
  /-
    case a
    K : Type v
    L : Type w
    inst✝¹ : Field K
    inst✝ : Field L
    i : RingHom K L
    x : K
    h : Polynomial K
    h_splits : Polynomial.Splits i h
    h_roots : Eq (Polynomial.map i h).roots (Singleton.singleton (i x))
    ⊢ Eq (Polynomial.map i h) (Polynomial.map i (HMul.hMul (Polynomial.C h.leading …
  -/
  rw [eq_prod_roots_of_splits h_splits, h_roots]
  /-
    case a
    K : Type v
    L : Type w
    inst✝¹ : Field K
    inst✝ : Field L
    i : RingHom K L
    x : K
    h : Polynomial K
    h_splits : Polynomial.Splits i h
    h_roots : Eq (Polynomial.map i h).roots (Singleton.singleton (i x))
    ⊢ Eq (HMul.hMul (Polynomial.C (i h.leadingCoeff)) (Multiset.map (fun a => HSub …
  -/
  simp
  /-
    🎉 no goals
  -/


variable (R) in
theorem mem_lift_of_splits_of_roots_mem_range [Algebra R K] {f : K[X]}
    (hs : f.Splits (RingHom.id K)) (hm : f.Monic) (hr : ∀ a ∈ f.roots, a ∈ (algebraMap R K).range) :
    f ∈ Polynomial.lifts (algebraMap R K) := by
  /-
    R : Type u_1
    K : Type v
    inst✝² : CommRing R
    inst✝¹ : Field K
    inst✝ : Algebra R K
    f : Polynomial K
    hs : Polynomial.Splits (RingHom.id K) f
    hm : f.Monic
    hr : ∀ (a : K), Membership.mem f.roots a → Membership.mem (algebraMap R K).ran …
    ⊢ Membership.mem (Polynomial.lifts (algebraMap R K)) f
  -/
  rw [eq_prod_roots_of_monic_of_splits_id hm hs, lifts_iff_liftsRing]
  /-
    R : Type u_1
    K : Type v
    inst✝² : CommRing R
    inst✝¹ : Field K
    inst✝ : Algebra R K
    f : Polynomial K
    hs : Polynomial.Splits (RingHom.id K) f
    hm : f.Monic
    hr : ∀ (a : K), Membership.mem f.roots a → Membership.mem (algebraMap R K).ran …
    ⊢ Membership.mem (Polynomial.liftsRing (algebraMap R K)) (Multiset.map (fun a  …
  -/
  refine Subring.multiset_prod_mem _ _ fun P hP => ?_
  /-
    R : Type u_1
    K : Type v
    inst✝² : CommRing R
    inst✝¹ : Field K
    inst✝ : Algebra R K
    f : Polynomial K
    hs : Polynomial.Splits (RingHom.id K) f
    hm : f.Monic
    hr : ∀ (a : K), Membership.mem f.roots a → Membership.mem (algebraMap R K).ran …
    P : Polynomial K
    hP : Membership.mem (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial …
    ⊢ Membership.mem (Polynomial.liftsRing (algebraMap R K)) P
  -/
  obtain ⟨b, hb, rfl⟩ := Multiset.mem_map.1 hP
  /-
    case intro.intro
    R : Type u_1
    K : Type v
    inst✝² : CommRing R
    inst✝¹ : Field K
    inst✝ : Algebra R K
    f : Polynomial K
    hs : Polynomial.Splits (RingHom.id K) f
    hm : f.Monic
    hr : ∀ (a : K), Membership.mem f.roots a → Membership.mem (algebraMap R K).ran …
    b : K
    hb : Membership.mem f.roots b
    hP : Membership.mem (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial …
    ⊢ Membership.mem (Polynomial.liftsRing (algebraMap R K)) (HSub.hSub Polynomial …
  -/
  exact Subring.sub_mem _ (X_mem_lifts _) (C'_mem_lifts (hr _ hb))
  /-
    🎉 no goals
  -/


local infixl:50 " ~ᵤ " => Associated


theorem splits_of_exists_multiset {f : K[X]} {s : Multiset L}
    (hs : f.map i = C (i f.leadingCoeff) * (s.map fun a : L => X - C a).prod) : Splits i f :=
  letI := Classical.decEq K
  if hf0 : f = 0 then hf0.symm ▸ splits_zero i
  else
    Or.inr @fun p hp hdp => by
      /-
        K : Type v
        L : Type w
        inst✝¹ : Field K
        inst✝ : Field L
        i : RingHom K L
        f : Polynomial K
        s : Multiset L
        hs : Eq (Polynomial.map i f) (HMul.hMul (Polynomial.C (i f.leadingCoeff)) (Mul …
        this : DecidableEq K := Classical.decEq K
        hf0 : Not (Eq f 0)
        p : Polynomial L
        hp : Irreducible p
        hdp : Dvd.dvd p (Polynomial.map i f)
        ⊢ Eq p.degree 1
      -/
      rw [irreducible_iff_prime] at hp
      /-
        K : Type v
        L : Type w
        inst✝¹ : Field K
        inst✝ : Field L
        i : RingHom K L
        f : Polynomial K
        s : Multiset L
        hs : Eq (Polynomial.map i f) (HMul.hMul (Polynomial.C (i f.leadingCoeff)) (Mul …
        this : DecidableEq K := Classical.decEq K
        hf0 : Not (Eq f 0)
        p : Polynomial L
        hp : Prime p
        hdp : Dvd.dvd p (Polynomial.map i f)
        ⊢ Eq p.degree 1
      -/
      rw [hs, ← Multiset.prod_toList] at hdp
      /-
        K : Type v
        L : Type w
        inst✝¹ : Field K
        inst✝ : Field L
        i : RingHom K L
        f : Polynomial K
        s : Multiset L
        hs : Eq (Polynomial.map i f) (HMul.hMul (Polynomial.C (i f.leadingCoeff)) (Mul …
        this : DecidableEq K := Classical.decEq K
        hf0 : Not (Eq f 0)
        p : Polynomial L
        hp : Prime p
        hdp : Dvd.dvd p (HMul.hMul (Polynomial.C (i f.leadingCoeff)) (Multiset.map (fu …
        ⊢ Eq p.degree 1
      -/
      obtain hd | hd := hp.2.2 _ _ hdp
        /-
          case inl
          K : Type v
          L : Type w
          inst✝¹ : Field K
          inst✝ : Field L
          i : RingHom K L
          f : Polynomial K
          s : Multiset L
          hs : Eq (Polynomial.map i f) (HMul.hMul (Polynomial.C (i f.leadingCoeff)) (Mul …
          this : DecidableEq K := Classical.decEq K
          hf0 : Not (Eq f 0)
          p : Polynomial L
          hp : Prime p
          hdp : Dvd.dvd p (HMul.hMul (Polynomial.C (i f.leadingCoeff)) (Multiset.map (fu …
          hd : Dvd.dvd p (Polynomial.C (i f.leadingCoeff))
          ⊢ Eq p.degree 1
        -/
      · refine (hp.2.1 <| isUnit_of_dvd_unit hd ?_).elim
        /-
          case inl
          K : Type v
          L : Type w
          inst✝¹ : Field K
          inst✝ : Field L
          i : RingHom K L
          f : Polynomial K
          s : Multiset L
          hs : Eq (Polynomial.map i f) (HMul.hMul (Polynomial.C (i f.leadingCoeff)) (Mul …
          this : DecidableEq K := Classical.decEq K
          hf0 : Not (Eq f 0)
          p : Polynomial L
          hp : Prime p
          hdp : Dvd.dvd p (HMul.hMul (Polynomial.C (i f.leadingCoeff)) (Multiset.map (fu …
          hd : Dvd.dvd p (Polynomial.C (i f.leadingCoeff))
          ⊢ IsUnit (Polynomial.C (i f.leadingCoeff))
        -/
        exact isUnit_C.2 ((leadingCoeff_ne_zero.2 hf0).isUnit.map i)
        /-
          🎉 no goals
        -/
        /-
          case inr
          K : Type v
          L : Type w
          inst✝¹ : Field K
          inst✝ : Field L
          i : RingHom K L
          f : Polynomial K
          s : Multiset L
          hs : Eq (Polynomial.map i f) (HMul.hMul (Polynomial.C (i f.leadingCoeff)) (Mul …
          this : DecidableEq K := Classical.decEq K
          hf0 : Not (Eq f 0)
          p : Polynomial L
          hp : Prime p
          hdp : Dvd.dvd p (HMul.hMul (Polynomial.C (i f.leadingCoeff)) (Multiset.map (fu …
          hd : Dvd.dvd p (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a) …
          ⊢ Eq p.degree 1
        -/
      · obtain ⟨q, hq, hd⟩ := hp.dvd_prod_iff.1 hd
        /-
          case inr.intro.intro
          K : Type v
          L : Type w
          inst✝¹ : Field K
          inst✝ : Field L
          i : RingHom K L
          f : Polynomial K
          s : Multiset L
          hs : Eq (Polynomial.map i f) (HMul.hMul (Polynomial.C (i f.leadingCoeff)) (Mul …
          this : DecidableEq K := Classical.decEq K
          hf0 : Not (Eq f 0)
          p : Polynomial L
          hp : Prime p
          hdp : Dvd.dvd p (HMul.hMul (Polynomial.C (i f.leadingCoeff)) (Multiset.map (fu …
          hd✝ : Dvd.dvd p (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a …
          q : Polynomial L
          hq : Membership.mem (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial …
          hd : Dvd.dvd p q
          ⊢ Eq p.degree 1
        -/
        obtain ⟨a, _, rfl⟩ := Multiset.mem_map.1 (Multiset.mem_toList.1 hq)
        /-
          case inr.intro.intro.intro.intro
          K : Type v
          L : Type w
          inst✝¹ : Field K
          inst✝ : Field L
          i : RingHom K L
          f : Polynomial K
          s : Multiset L
          hs : Eq (Polynomial.map i f) (HMul.hMul (Polynomial.C (i f.leadingCoeff)) (Mul …
          this : DecidableEq K := Classical.decEq K
          hf0 : Not (Eq f 0)
          p : Polynomial L
          hp : Prime p
          hdp : Dvd.dvd p (HMul.hMul (Polynomial.C (i f.leadingCoeff)) (Multiset.map (fu …
          hd✝ : Dvd.dvd p (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a …
          a : L
          left✝ : Membership.mem s a
          hq : Membership.mem (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial …
          hd : Dvd.dvd p (HSub.hSub Polynomial.X (Polynomial.C a))
          ⊢ Eq p.degree 1
        -/
        rw [degree_eq_degree_of_associated ((hp.dvd_prime_iff_associated <| prime_X_sub_C a).1 hd)]
        /-
          case inr.intro.intro.intro.intro
          K : Type v
          L : Type w
          inst✝¹ : Field K
          inst✝ : Field L
          i : RingHom K L
          f : Polynomial K
          s : Multiset L
          hs : Eq (Polynomial.map i f) (HMul.hMul (Polynomial.C (i f.leadingCoeff)) (Mul …
          this : DecidableEq K := Classical.decEq K
          hf0 : Not (Eq f 0)
          p : Polynomial L
          hp : Prime p
          hdp : Dvd.dvd p (HMul.hMul (Polynomial.C (i f.leadingCoeff)) (Multiset.map (fu …
          hd✝ : Dvd.dvd p (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a …
          a : L
          left✝ : Membership.mem s a
          hq : Membership.mem (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial …
          hd : Dvd.dvd p (HSub.hSub Polynomial.X (Polynomial.C a))
          ⊢ Eq (HSub.hSub Polynomial.X (Polynomial.C a)).degree 1
        -/
        exact degree_X_sub_C a
        /-
          🎉 no goals
        -/


theorem splits_of_splits_id {f : K[X]} : Splits (RingHom.id K) f → Splits i f :=
  UniqueFactorizationMonoid.induction_on_prime f (fun _ => splits_zero _)
                                                                                         /-
                                                                                           K : Type v
                                                                                           L : Type w
                                                                                           inst✝¹ : Field K
                                                                                           inst✝ : Field L
                                                                                           i : RingHom K L
                                                                                           f x✝¹ : Polynomial K
                                                                                           hu : IsUnit x✝¹
                                                                                           x✝ : Polynomial.Splits (RingHom.id K) x✝¹
                                                                                           ⊢ LE.le 0 1
                                                                                         -/
    (fun _ hu _ => splits_of_degree_le_one _ ((isUnit_iff_degree_eq_zero.1 hu).symm ▸ by decide))
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
    fun _ p ha0 hp ih hfi =>
    splits_mul _
      (splits_of_degree_eq_one _
        ((splits_of_splits_mul _ (mul_ne_zero hp.1 ha0) hfi).1.def.resolve_left hp.1 hp.irreducible
              /-
                K : Type v
                L : Type w
                inst✝¹ : Field K
                inst✝ : Field L
                i : RingHom K L
                f x✝ p : Polynomial K
                ha0 : Ne x✝ 0
                hp : Prime p
                ih : Polynomial.Splits (RingHom.id K) x✝ → Polynomial.Splits i x✝
                hfi : Polynomial.Splits (RingHom.id K) (HMul.hMul p x✝)
                ⊢ Dvd.dvd p (Polynomial.map (RingHom.id K) p)
              -/
          (by rw [map_id])))
              /-
                🎉 no goals
              -/
      (ih (splits_of_splits_mul _ (mul_ne_zero hp.1 ha0) hfi).2)


theorem splits_iff_exists_multiset {f : K[X]} :
    Splits i f ↔
      ∃ s : Multiset L, f.map i = C (i f.leadingCoeff) * (s.map fun a : L => X - C a).prod :=
  ⟨fun hf => ⟨(f.map i).roots, eq_prod_roots_of_splits hf⟩, fun ⟨_, hs⟩ =>
    splits_of_exists_multiset i hs⟩


theorem splits_of_comp (j : L →+* F) {f : K[X]} (h : Splits (j.comp i) f)
    (roots_mem_range : ∀ a ∈ (f.map (j.comp i)).roots, a ∈ j.range) : Splits i f := by
  /-
    F : Type u
    K : Type v
    L : Type w
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Field F
    i : RingHom K L
    j : RingHom L F
    f : Polynomial K
    h : Polynomial.Splits (j.comp i) f
    roots_mem_range : ∀ (a : F), Membership.mem (Polynomial.map (j.comp i) f).root …
    ⊢ Polynomial.Splits i f
  -/
  choose lift lift_eq using roots_mem_range
  /-
    F : Type u
    K : Type v
    L : Type w
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Field F
    i : RingHom K L
    j : RingHom L F
    f : Polynomial K
    h : Polynomial.Splits (j.comp i) f
    lift : (a : F) → Membership.mem (Polynomial.map (j.comp i) f).roots a → L
    lift_eq : ∀ (a : F) (a_1 : Membership.mem (Polynomial.map (j.comp i) f).roots  …
    ⊢ Polynomial.Splits i f
  -/
  rw [splits_iff_exists_multiset]
  /-
    F : Type u
    K : Type v
    L : Type w
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Field F
    i : RingHom K L
    j : RingHom L F
    f : Polynomial K
    h : Polynomial.Splits (j.comp i) f
    lift : (a : F) → Membership.mem (Polynomial.map (j.comp i) f).roots a → L
    lift_eq : ∀ (a : F) (a_1 : Membership.mem (Polynomial.map (j.comp i) f).roots  …
    ⊢ Exists fun s => Eq (Polynomial.map i f) (HMul.hMul (Polynomial.C (i f.leadin …
  -/
  refine ⟨(f.map (j.comp i)).roots.pmap lift fun _ ↦ id, map_injective _ j.injective ?_⟩
  /-
    F : Type u
    K : Type v
    L : Type w
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Field F
    i : RingHom K L
    j : RingHom L F
    f : Polynomial K
    h : Polynomial.Splits (j.comp i) f
    lift : (a : F) → Membership.mem (Polynomial.map (j.comp i) f).roots a → L
    lift_eq : ∀ (a : F) (a_1 : Membership.mem (Polynomial.map (j.comp i) f).roots  …
    ⊢ Eq (Polynomial.map j (Polynomial.map i f)) (Polynomial.map j (HMul.hMul (Pol …
  -/
  conv_lhs => rw [Polynomial.map_map, eq_prod_roots_of_splits h]
  simp_rw [Polynomial.map_mul, Polynomial.map_multiset_prod, Multiset.map_pmap, Polynomial.map_sub,
    map_C, map_X, lift_eq, Multiset.pmap_eq_map]
  /-
    F : Type u
    K : Type v
    L : Type w
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Field F
    i : RingHom K L
    j : RingHom L F
    f : Polynomial K
    h : Polynomial.Splits (j.comp i) f
    lift : (a : F) → Membership.mem (Polynomial.map (j.comp i) f).roots a → L
    lift_eq : ∀ (a : F) (a_1 : Membership.mem (Polynomial.map (j.comp i) f).roots  …
    ⊢ Eq (HMul.hMul (Polynomial.C ((j.comp i) f.leadingCoeff)) (Multiset.map (fun  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem splits_id_of_splits {f : K[X]} (h : Splits i f)
    (roots_mem_range : ∀ a ∈ (f.map i).roots, a ∈ i.range) : Splits (RingHom.id K) f :=
  splits_of_comp (RingHom.id K) i h roots_mem_range


theorem splits_comp_of_splits (i : R →+* K) (j : K →+* L) {f : R[X]} (h : Splits i f) :
    Splits (j.comp i) f :=
  (splits_map_iff i j).mp (splits_of_splits_id _ <| (splits_map_iff i <| .id K).mpr h)


theorem splits_of_algHom {f : R[X]} (h : Splits (algebraMap R K) f) (e : K →ₐ[R] L) :
    Splits (algebraMap R L) f := by
  /-
    R : Type u_1
    K : Type v
    L : Type w
    inst✝⁴ : CommRing R
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra R K
    inst✝ : Algebra R L
    f : Polynomial R
    h : Polynomial.Splits (algebraMap R K) f
    e : AlgHom R K L
    ⊢ Polynomial.Splits (algebraMap R L) f
  -/
  rw [← e.comp_algebraMap_of_tower R]; exact splits_comp_of_splits _ _ h
                                       /-
                                         🎉 no goals
                                       -/


variable (L) in
theorem splits_of_isScalarTower {f : R[X]} [Algebra K L] [IsScalarTower R K L]
    (h : Splits (algebraMap R K) f) : Splits (algebraMap R L) f :=
  splits_of_algHom h (IsScalarTower.toAlgHom R K L)


/-- A polynomial splits if and only if it has as many roots as its degree. -/
theorem splits_iff_card_roots {p : K[X]} :
    Splits (RingHom.id K) p ↔ Multiset.card p.roots = p.natDegree := by
  /-
    K : Type v
    inst✝ : Field K
    p : Polynomial K
    ⊢ Iff (Polynomial.Splits (RingHom.id K) p) (Eq p.roots.card p.natDegree)
  -/
  constructor
    /-
      case mp
      K : Type v
      inst✝ : Field K
      p : Polynomial K
      ⊢ Polynomial.Splits (RingHom.id K) p → Eq p.roots.card p.natDegree
    -/
  · intro H
    /-
      case mp
      K : Type v
      inst✝ : Field K
      p : Polynomial K
      H : Polynomial.Splits (RingHom.id K) p
      ⊢ Eq p.roots.card p.natDegree
    -/
    rw [natDegree_eq_card_roots H, map_id]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type v
      inst✝ : Field K
      p : Polynomial K
      ⊢ Eq p.roots.card p.natDegree → Polynomial.Splits (RingHom.id K) p
    -/
  · intro hroots
    /-
      case mpr
      K : Type v
      inst✝ : Field K
      p : Polynomial K
      hroots : Eq p.roots.card p.natDegree
      ⊢ Polynomial.Splits (RingHom.id K) p
    -/
    rw [splits_iff_exists_multiset (RingHom.id K)]
    /-
      case mpr
      K : Type v
      inst✝ : Field K
      p : Polynomial K
      hroots : Eq p.roots.card p.natDegree
      ⊢ Exists fun s => Eq (Polynomial.map (RingHom.id K) p) (HMul.hMul (Polynomial. …
    -/
    use p.roots
    /-
      case h
      K : Type v
      inst✝ : Field K
      p : Polynomial K
      hroots : Eq p.roots.card p.natDegree
      ⊢ Eq (Polynomial.map (RingHom.id K) p) (HMul.hMul (Polynomial.C ((RingHom.id K …
    -/
    simp only [RingHom.id_apply, map_id]
    /-
      case h
      K : Type v
      inst✝ : Field K
      p : Polynomial K
      hroots : Eq p.roots.card p.natDegree
      ⊢ Eq p (HMul.hMul (Polynomial.C p.leadingCoeff) (Multiset.map (fun a => HSub.h …
    -/
    exact (C_leadingCoeff_mul_prod_multiset_X_sub_C hroots).symm
    /-
      🎉 no goals
    -/


theorem aeval_root_derivative_of_splits [Algebra K L] [DecidableEq L] {P : K[X]} (hmo : P.Monic)
    (hP : P.Splits (algebraMap K L)) {r : L} (hr : r ∈ P.aroots L) :
    aeval r (Polynomial.derivative P) =
    (((P.aroots L).erase r).map fun a => r - a).prod := by
  /-
    K : Type v
    L : Type w
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : DecidableEq L
    P : Polynomial K
    hmo : P.Monic
    hP : Polynomial.Splits (algebraMap K L) P
    r : L
    hr : Membership.mem (P.aroots L) r
    ⊢ Eq ((Polynomial.aeval r) (Polynomial.derivative P)) (Multiset.map (fun a =>  …
  -/
  replace hmo := hmo.map (algebraMap K L)
  /-
    K : Type v
    L : Type w
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : DecidableEq L
    P : Polynomial K
    hP : Polynomial.Splits (algebraMap K L) P
    r : L
    hr : Membership.mem (P.aroots L) r
    hmo : (Polynomial.map (algebraMap K L) P).Monic
    ⊢ Eq ((Polynomial.aeval r) (Polynomial.derivative P)) (Multiset.map (fun a =>  …
  -/
  replace hP := (splits_id_iff_splits (algebraMap K L)).2 hP
  /-
    K : Type v
    L : Type w
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : DecidableEq L
    P : Polynomial K
    r : L
    hr : Membership.mem (P.aroots L) r
    hmo : (Polynomial.map (algebraMap K L) P).Monic
    hP : Polynomial.Splits (RingHom.id L) (Polynomial.map (algebraMap K L) P)
    ⊢ Eq ((Polynomial.aeval r) (Polynomial.derivative P)) (Multiset.map (fun a =>  …
  -/
  rw [aeval_def, ← eval_map, ← derivative_map]
  /-
    K : Type v
    L : Type w
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : DecidableEq L
    P : Polynomial K
    r : L
    hr : Membership.mem (P.aroots L) r
    hmo : (Polynomial.map (algebraMap K L) P).Monic
    hP : Polynomial.Splits (RingHom.id L) (Polynomial.map (algebraMap K L) P)
    ⊢ Eq (Polynomial.eval r (Polynomial.derivative (Polynomial.map (algebraMap K L …
  -/
  nth_rw 1 [eq_prod_roots_of_monic_of_splits_id hmo hP]
  /-
    K : Type v
    L : Type w
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : DecidableEq L
    P : Polynomial K
    r : L
    hr : Membership.mem (P.aroots L) r
    hmo : (Polynomial.map (algebraMap K L) P).Monic
    hP : Polynomial.Splits (RingHom.id L) (Polynomial.map (algebraMap K L) P)
    ⊢ Eq (Polynomial.eval r (Polynomial.derivative (Multiset.map (fun a => HSub.hS …
  -/
  rw [eval_multiset_prod_X_sub_C_derivative hr]
  /-
    🎉 no goals
  -/


/-- If `P` is a monic polynomial that splits, then `coeff P 0` equals the product of the roots. -/
theorem prod_roots_eq_coeff_zero_of_monic_of_splits {P : K[X]} (hmo : P.Monic)
    (hP : P.Splits (RingHom.id K)) : coeff P 0 = (-1) ^ P.natDegree * P.roots.prod := by
  /-
    K : Type v
    inst✝ : Field K
    P : Polynomial K
    hmo : P.Monic
    hP : Polynomial.Splits (RingHom.id K) P
    ⊢ Eq (P.coeff 0) (HMul.hMul (HPow.hPow (-1) P.natDegree) P.roots.prod)
  -/
  nth_rw 1 [eq_prod_roots_of_monic_of_splits_id hmo hP]
  /-
    K : Type v
    inst✝ : Field K
    P : Polynomial K
    hmo : P.Monic
    hP : Polynomial.Splits (RingHom.id K) P
    ⊢ Eq ((Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) P.roots …
  -/
  rw [coeff_zero_eq_eval_zero, eval_multiset_prod, Multiset.map_map]
  /-
    K : Type v
    inst✝ : Field K
    P : Polynomial K
    hmo : P.Monic
    hP : Polynomial.Splits (RingHom.id K) P
    ⊢ Eq (Multiset.map (Function.comp (Polynomial.eval 0) fun a => HSub.hSub Polyn …
  -/
  simp_rw [Function.comp_apply, eval_sub, eval_X, zero_sub, eval_C]
  conv_lhs =>
    congr
    congr
    ext
    rw [neg_eq_neg_one_mul]
  /-
    K : Type v
    inst✝ : Field K
    P : Polynomial K
    hmo : P.Monic
    hP : Polynomial.Splits (RingHom.id K) P
    ⊢ Eq (Multiset.map (HMul.hMul (-1)) P.roots).prod (HMul.hMul (HPow.hPow (-1) P …
  -/
  simp only [splits_iff_card_roots.1 hP, neg_mul, one_mul, Multiset.prod_map_neg]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-01")]
alias prod_roots_eq_coeff_zero_of_monic_of_split := prod_roots_eq_coeff_zero_of_monic_of_splits


/-- If `P` is a monic polynomial that splits, then `P.nextCoeff` equals the sum of the roots. -/
theorem sum_roots_eq_nextCoeff_of_monic_of_split {P : K[X]} (hmo : P.Monic)
    (hP : P.Splits (RingHom.id K)) : P.nextCoeff = -P.roots.sum := by
  /-
    K : Type v
    inst✝ : Field K
    P : Polynomial K
    hmo : P.Monic
    hP : Polynomial.Splits (RingHom.id K) P
    ⊢ Eq P.nextCoeff (Neg.neg P.roots.sum)
  -/
  nth_rw 1 [eq_prod_roots_of_monic_of_splits_id hmo hP]
  /-
    K : Type v
    inst✝ : Field K
    P : Polynomial K
    hmo : P.Monic
    hP : Polynomial.Splits (RingHom.id K) P
    ⊢ Eq (Multiset.map (fun a => HSub.hSub Polynomial.X (Polynomial.C a)) P.roots) …
  -/
  rw [Monic.nextCoeff_multiset_prod _ _ fun a ha => _]
    /-
      K : Type v
      inst✝ : Field K
      P : Polynomial K
      hmo : P.Monic
      hP : Polynomial.Splits (RingHom.id K) P
      ⊢ Eq (Multiset.map (fun i => (HSub.hSub Polynomial.X (Polynomial.C i)).nextCoe …
    -/
  · simp_rw [nextCoeff_X_sub_C, Multiset.sum_map_neg']
    /-
      🎉 no goals
    -/
    /-
      K : Type v
      inst✝ : Field K
      P : Polynomial K
      hmo : P.Monic
      hP : Polynomial.Splits (RingHom.id K) P
      ⊢ ∀ (a : K), Membership.mem P.roots a → (HSub.hSub Polynomial.X (Polynomial.C  …
    -/
  · simp only [monic_X_sub_C, implies_true]
    /-
      🎉 no goals
    -/


