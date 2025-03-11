@[simp]
theorem eval₂_at_zero : p.eval₂ f 0 = f (coeff p 0) := by
  simp +contextual only [eval₂_eq_sum, zero_pow_eq, mul_ite, mul_zero,
    mul_one, sum, Classical.not_not, mem_support_iff, sum_ite_eq', ite_eq_left_iff,
    RingHom.map_zero, imp_true_iff, eq_self_iff_true]


@[simp]
theorem eval₂_C_X : eval₂ C X p = p :=
                                                  /-
                                                    R : Type u
                                                    inst✝ : Semiring R
                                                    p✝ p q : Polynomial R
                                                    hp : Eq (Polynomial.eval₂ Polynomial.C Polynomial.X p) p
                                                    hq : Eq (Polynomial.eval₂ Polynomial.C Polynomial.X q) q
                                                    ⊢ Eq (Polynomial.eval₂ Polynomial.C Polynomial.X (HAdd.hAdd p q)) (HAdd.hAdd p …
                                                  -/
  Polynomial.induction_on' p (fun p q hp hq => by simp [hp, hq]) fun n x => by
                                                  /-
                                                    🎉 no goals
                                                  -/
    /-
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      x : R
      ⊢ Eq (Polynomial.eval₂ Polynomial.C Polynomial.X ((Polynomial.monomial n) x))  …
    -/
    rw [eval₂_monomial, ← smul_X_eq_monomial, C_mul']
    /-
      🎉 no goals
    -/


theorem coeff_zero_eq_eval_zero (p : R[X]) : coeff p 0 = p.eval 0 :=
  calc
                                        /-
                                          R : Type u
                                          inst✝ : Semiring R
                                          p : Polynomial R
                                          ⊢ Eq (p.coeff 0) (HMul.hMul (p.coeff 0) (HPow.hPow 0 0))
                                        -/
    coeff p 0 = coeff p 0 * 0 ^ 0 := by simp
                                        /-
                                          🎉 no goals
                                        -/
    _ = p.eval 0 := by
      /-
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        ⊢ Eq (HMul.hMul (p.coeff 0) (HPow.hPow 0 0)) (Polynomial.eval 0 p)
      -/
      symm
      /-
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        ⊢ Eq (Polynomial.eval 0 p) (HMul.hMul (p.coeff 0) (HPow.hPow 0 0))
      -/
      rw [eval_eq_sum]
      /-
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        ⊢ Eq (p.sum fun e a => HMul.hMul a (HPow.hPow 0 e)) (HMul.hMul (p.coeff 0) (HP …
      -/
      exact Finset.sum_eq_single _ (fun b _ hb => by simp [zero_pow hb]) (by simp)
      /-
        🎉 no goals
      -/


theorem zero_isRoot_of_coeff_zero_eq_zero {p : R[X]} (hp : p.coeff 0 = 0) : IsRoot p 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Eq (p.coeff 0) 0
    ⊢ p.IsRoot 0
  -/
  rwa [coeff_zero_eq_eval_zero] at hp
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_map (n : ℕ) : coeff (p.map f) n = f (coeff p n) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    n : Nat
    ⊢ Eq ((Polynomial.map f p).coeff n) (f (p.coeff n))
  -/
  rw [map, eval₂_def, coeff_sum, sum]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    n : Nat
    ⊢ Eq (p.support.sum fun n_1 => (HMul.hMul ((Polynomial.C.comp f) (p.coeff n_1) …
  -/
  conv_rhs => rw [← sum_C_mul_X_pow_eq p, coeff_sum, sum, map_sum]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    n : Nat
    ⊢ Eq (p.support.sum fun n_1 => (HMul.hMul ((Polynomial.C.comp f) (p.coeff n_1) …
  -/
  refine Finset.sum_congr rfl fun x _hx => ?_
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    n x : Nat
    _hx : Membership.mem p.support x
    ⊢ Eq ((HMul.hMul ((Polynomial.C.comp f) (p.coeff x)) (HPow.hPow Polynomial.X x …
  -/
  simp only [RingHom.coe_comp, Function.comp, coeff_C_mul_X_pow]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    n x : Nat
    _hx : Membership.mem p.support x
    ⊢ Eq (ite (Eq n x) (f (p.coeff x)) 0) (f (ite (Eq n x) (p.coeff x) 0))
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [f.map_zero]
                /-
                  🎉 no goals
                -/


lemma coeff_map_eq_comp (p : R[X]) (f : R →+* S) : (p.map f).coeff = f ∘ p.coeff := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    ⊢ Eq (Polynomial.map f p).coeff (Function.comp (⇑f) p.coeff)
  -/
  ext n; exact coeff_map ..
         /-
           🎉 no goals
         -/


theorem map_map [Semiring T] (g : S →+* T) (p : R[X]) : (p.map f).map g = p.map (g.comp f) :=
          /-
            R : Type u
            S : Type v
            T : Type w
            inst✝² : Semiring R
            inst✝¹ : Semiring S
            f : RingHom R S
            inst✝ : Semiring T
            g : RingHom S T
            p : Polynomial R
            ⊢ ∀ (n : Nat), Eq ((Polynomial.map g (Polynomial.map f p)).coeff n) ((Polynomi …
          -/
  ext (by simp [coeff_map])
          /-
            🎉 no goals
          -/


@[simp]
                                                /-
                                                  R : Type u
                                                  inst✝ : Semiring R
                                                  p : Polynomial R
                                                  ⊢ Eq (Polynomial.map (RingHom.id R) p) p
                                                -/
theorem map_id : p.map (RingHom.id _) = p := by simp [Polynomial.ext_iff, coeff_map]
                                                /-
                                                  🎉 no goals
                                                -/


/-- The polynomial ring over a finite product of rings is isomorphic to
the product of polynomial rings over individual rings. -/
def piEquiv {ι} [Finite ι] (R : ι → Type*) [∀ i, Semiring (R i)] :
    (∀ i, R i)[X] ≃+* ∀ i, (R i)[X] :=
  .ofBijective (Pi.ringHom fun i ↦ mapRingHom (Pi.evalRingHom R i))
                    /-
                      R✝ : Type u
                      S : Type v
                      T : Type w
                      ι✝ : Type y
                      a b : R✝
                      m n : Nat
                      inst✝³ : Semiring R✝
                      p✝ q✝ r : Polynomial R✝
                      inst✝² : Semiring S
                      f : RingHom R✝ S
                      ι : Type ?u.12619
                      inst✝¹ : Finite ι
                      R : ι → Type u_1
                      inst✝ : (i : ι) → Semiring (R i)
                      p q : Polynomial ((i : ι) → R i)
                      h : Eq ((Pi.ringHom fun i => Polynomial.mapRingHom (Pi.evalRingHom R i)) p) (( …
                      ⊢ Eq p q
                    -/
    ⟨fun p q h ↦ by ext n i; simpa using congr_arg (fun p ↦ coeff (p i) n) h,
                             /-
                               🎉 no goals
                             -/
      fun p ↦ ⟨.ofFinsupp (.ofSupportFinite (fun n i ↦ coeff (p i) n) <|
        (Set.finite_iUnion fun i ↦ (p i).support.finite_toSet).subset fun n hn ↦ by
          /-
            R✝ : Type u
            S : Type v
            T : Type w
            ι✝ : Type y
            a b : R✝
            m n✝ : Nat
            inst✝³ : Semiring R✝
            p✝ q r : Polynomial R✝
            inst✝² : Semiring S
            f : RingHom R✝ S
            ι : Type ?u.12619
            inst✝¹ : Finite ι
            R : ι → Type u_1
            inst✝ : (i : ι) → Semiring (R i)
            p : (i : ι) → Polynomial (R i)
            n : Nat
            hn : Membership.mem (Function.support fun n i => (p i).coeff n) n
            ⊢ Membership.mem (Set.iUnion fun i => ↑(p i).support) n
          -/
          simp only [Set.mem_iUnion, Finset.mem_coe, mem_support_iff, Function.mem_support] at hn ⊢
          /-
            R✝ : Type u
            S : Type v
            T : Type w
            ι✝ : Type y
            a b : R✝
            m n✝ : Nat
            inst✝³ : Semiring R✝
            p✝ q r : Polynomial R✝
            inst✝² : Semiring S
            f : RingHom R✝ S
            ι : Type ?u.12619
            inst✝¹ : Finite ι
            R : ι → Type u_1
            inst✝ : (i : ι) → Semiring (R i)
            p : (i : ι) → Polynomial (R i)
            n : Nat
            hn : Ne (fun i => (p i).coeff n) 0
            ⊢ Exists fun i => Ne ((p i).coeff n) 0
          -/
                          /-
                            🎉 no goals
                          -/
          contrapose! hn; exact funext hn), by ext i n; exact coeff_map _ _⟩⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem map_injective (hf : Function.Injective f) : Function.Injective (map f) := fun p q h =>
                        /-
                          R : Type u
                          S : Type v
                          inst✝¹ : Semiring R
                          inst✝ : Semiring S
                          f : RingHom R S
                          hf : Function.Injective ⇑f
                          p q : Polynomial R
                          h : Eq (Polynomial.map f p) (Polynomial.map f q)
                          m : Nat
                          ⊢ Eq (f (p.coeff m)) (f (q.coeff m))
                        -/
  ext fun m => hf <| by rw [← coeff_map f, ← coeff_map f, h]
                        /-
                          🎉 no goals
                        -/


theorem map_surjective (hf : Function.Surjective f) : Function.Surjective (map f) := fun p =>
  Polynomial.induction_on' p
    (fun p q hp hq =>
      let ⟨p', hp'⟩ := hp
      let ⟨q', hq'⟩ := hq
                   /-
                     R : Type u
                     S : Type v
                     inst✝¹ : Semiring R
                     inst✝ : Semiring S
                     f : RingHom R S
                     hf : Function.Surjective ⇑f
                     p✝ p q : Polynomial S
                     hp : Exists fun a => Eq (Polynomial.map f a) p
                     hq : Exists fun a => Eq (Polynomial.map f a) q
                     p' : Polynomial R
                     hp' : Eq (Polynomial.map f p') p
                     q' : Polynomial R
                     hq' : Eq (Polynomial.map f q') q
                     ⊢ Eq (Polynomial.map f (HAdd.hAdd p' q')) (HAdd.hAdd p q)
                   -/
      ⟨p' + q', by rw [Polynomial.map_add f, hp', hq']⟩)
                   /-
                     🎉 no goals
                   -/
    fun n s =>
    let ⟨r, hr⟩ := hf s
                      /-
                        R : Type u
                        S : Type v
                        inst✝¹ : Semiring R
                        inst✝ : Semiring S
                        f : RingHom R S
                        hf : Function.Surjective ⇑f
                        p : Polynomial S
                        n : Nat
                        s : S
                        r : R
                        hr : Eq (f r) s
                        ⊢ Eq (Polynomial.map f ((Polynomial.monomial n) r)) ((Polynomial.monomial n) s)
                      -/
    ⟨monomial n r, by rw [map_monomial f, hr]⟩
                      /-
                        🎉 no goals
                      -/


protected theorem map_eq_zero_iff (hf : Function.Injective f) : p.map f = 0 ↔ p = 0 :=
  map_eq_zero_iff (mapRingHom f) (map_injective f hf)


protected theorem map_ne_zero_iff (hf : Function.Injective f) : p.map f ≠ 0 ↔ p ≠ 0 :=
  (Polynomial.map_eq_zero_iff hf).not


@[simp]
theorem mapRingHom_id : mapRingHom (RingHom.id R) = RingHom.id R[X] :=
  RingHom.ext fun _x => map_id


@[simp]
theorem mapRingHom_comp [Semiring T] (f : S →+* T) (g : R →+* S) :
    (mapRingHom f).comp (mapRingHom g) = mapRingHom (f.comp g) :=
  RingHom.ext <| Polynomial.map_map g f


theorem eval₂_map [Semiring T] (g : S →+* T) (x : T) :
    (p.map f).eval₂ g x = p.eval₂ (g.comp f) x := by
  /-
    R : Type u
    S : Type v
    T : Type w
    inst✝² : Semiring R
    p : Polynomial R
    inst✝¹ : Semiring S
    f : RingHom R S
    inst✝ : Semiring T
    g : RingHom S T
    x : T
    ⊢ Eq (Polynomial.eval₂ g x (Polynomial.map f p)) (Polynomial.eval₂ (g.comp f)  …
  -/
  rw [eval₂_eq_eval_map, eval₂_eq_eval_map, map_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem eval_zero_map (f : R →+* S) (p : R[X]) : (p.map f).eval 0 = f (p.eval 0) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    ⊢ Eq (Polynomial.eval 0 (Polynomial.map f p)) (f (Polynomial.eval 0 p))
  -/
  simp [← coeff_zero_eq_eval_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem eval_one_map (f : R →+* S) (p : R[X]) : (p.map f).eval 1 = f (p.eval 1) := by
  induction p using Polynomial.induction_on' with
  | h_add p q hp hq =>
    simp only [hp, hq, Polynomial.map_add, RingHom.map_add, eval_add]
  | h_monomial n r =>
    simp only [one_pow, mul_one, eval_monomial, map_monomial]


@[simp]
theorem eval_natCast_map (f : R →+* S) (p : R[X]) (n : ℕ) :
    (p.map f).eval (n : S) = f (p.eval n) := by
  induction p using Polynomial.induction_on' with
  | h_add p q hp hq =>
    simp only [hp, hq, Polynomial.map_add, RingHom.map_add, eval_add]
  | h_monomial n r =>
    simp only [map_natCast f, eval_monomial, map_monomial, f.map_pow, f.map_mul]


@[deprecated (since := "2024-04-17")]
alias eval_nat_cast_map := eval_natCast_map


@[simp]
theorem eval_intCast_map {R S : Type*} [Ring R] [Ring S] (f : R →+* S) (p : R[X]) (i : ℤ) :
    (p.map f).eval (i : S) = f (p.eval i) := by
  induction p using Polynomial.induction_on' with
  | h_add p q hp hq =>
    simp only [hp, hq, Polynomial.map_add, RingHom.map_add, eval_add]
  | h_monomial n r =>
    simp only [map_intCast, eval_monomial, map_monomial, map_pow, map_mul]


@[deprecated (since := "2024-04-17")]
alias eval_int_cast_map := eval_intCast_map


theorem hom_eval₂ (x : S) : g (p.eval₂ f x) = p.eval₂ (g.comp f) (g x) := by
  /-
    R : Type u
    S : Type v
    T : Type w
    inst✝² : Semiring R
    p : Polynomial R
    inst✝¹ : Semiring S
    inst✝ : Semiring T
    f : RingHom R S
    g : RingHom S T
    x : S
    ⊢ Eq (g (Polynomial.eval₂ f x p)) (Polynomial.eval₂ (g.comp f) (g x) p)
  -/
  rw [← eval₂_map, eval₂_at_apply, eval_map]
  /-
    🎉 no goals
  -/


theorem eval₂_hom (x : R) : p.eval₂ f (f x) = f (p.eval x) :=
  RingHom.comp_id f ▸ (hom_eval₂ p (RingHom.id R) f x).symm


theorem evalRingHom_zero : evalRingHom 0 = constantCoeff :=
  DFunLike.ext _ _ fun p => p.coeff_zero_eq_eval_zero.symm


theorem support_map_subset [Semiring R] [Semiring S] (f : R →+* S) (p : R[X]) :
    (map f p).support ⊆ p.support := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    ⊢ HasSubset.Subset (Polynomial.map f p).support p.support
  -/
  intro x
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    x : Nat
    ⊢ Membership.mem (Polynomial.map f p).support x → Membership.mem p.support x
  -/
  contrapose!
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    x : Nat
    ⊢ Not (Membership.mem p.support x) → Not (Membership.mem (Polynomial.map f p). …
  -/
  simp +contextual
  /-
    🎉 no goals
  -/


theorem support_map_of_injective [Semiring R] [Semiring S] (p : R[X]) {f : R →+* S}
    (hf : Function.Injective f) : (map f p).support = p.support := by
  simp_rw [Finset.ext_iff, mem_support_iff, coeff_map, ← map_zero f, hf.ne_iff,
    forall_const]


theorem IsRoot.map {f : R →+* S} {x : R} {p : R[X]} (h : IsRoot p x) : IsRoot (p.map f) (f x) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    x : R
    p : Polynomial R
    h : p.IsRoot x
    ⊢ (Polynomial.map f p).IsRoot (f x)
  -/
  rw [IsRoot, eval_map, eval₂_hom, h.eq_zero, f.map_zero]
  /-
    🎉 no goals
  -/


theorem IsRoot.of_map {R} [CommRing R] {f : R →+* S} {x : R} {p : R[X]} (h : IsRoot (p.map f) (f x))
    (hf : Function.Injective f) : IsRoot p x := by
  /-
    S : Type v
    inst✝¹ : CommSemiring S
    R : Type u_1
    inst✝ : CommRing R
    f : RingHom R S
    x : R
    p : Polynomial R
    h : (Polynomial.map f p).IsRoot (f x)
    hf : Function.Injective ⇑f
    ⊢ p.IsRoot x
  -/
  rwa [IsRoot, ← (injective_iff_map_eq_zero' f).mp hf, ← eval₂_hom, ← eval_map]
  /-
    🎉 no goals
  -/


theorem isRoot_map_iff {R : Type*} [CommRing R] {f : R →+* S} {x : R} {p : R[X]}
    (hf : Function.Injective f) : IsRoot (p.map f) (f x) ↔ IsRoot p x :=
  ⟨fun h => h.of_map hf, fun h => h.map⟩


