local notation:max R "<" x:max ">" => adjoin R ({x} : Set S)


/-- Let `S / R` be a ring extension and `x : S`, then the conductor of `R<x>` is the
    biggest ideal of `S` contained in `R<x>`. -/
def conductor (x : S) : Ideal S where
  carrier := {a | ∀ b : S, a * b ∈ R<x>}
                    /-
                      R : Type u_1
                      S : Type u_2
                      inst✝² : CommRing R
                      inst✝¹ : CommRing S
                      inst✝ : Algebra R S
                      x b : S
                      ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton x)) (HMul.hMul 0 b)
                    -/
                         /-
                           R : Type u_1
                           S : Type u_2
                           inst✝² : CommRing R
                           inst✝¹ : CommRing S
                           inst✝ : Algebra R S
                           x a✝ b✝ : S
                           ha : Membership.mem (setOf fun a => ∀ (b : S), Membership.mem (Algebra.adjoin  …
                           hb : Membership.mem (setOf fun a => ∀ (b : S), Membership.mem (Algebra.adjoin  …
                           c : S
                           ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton x)) (HMul.hMul (HAdd.h …
                         -/
  zero_mem' b := by simpa only [zero_mul] using Subalgebra.zero_mem _
                         /-
                           🎉 no goals
                         -/
                    /-
                      🎉 no goals
                    -/
  add_mem' ha hb c := by simpa only [add_mul] using Subalgebra.add_mem _ (ha c) (hb c)
                           /-
                             R : Type u_1
                             S : Type u_2
                             inst✝² : CommRing R
                             inst✝¹ : CommRing S
                             inst✝ : Algebra R S
                             x c a : S
                             ha : Membership.mem { carrier := setOf fun a => ∀ (b : S), Membership.mem (Alg …
                             b : S
                             ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton x)) (HMul.hMul (HSMul. …
                           -/
  smul_mem' c a ha b := by simpa only [smul_eq_mul, mul_left_comm, mul_assoc] using ha (c * b)
                           /-
                             🎉 no goals
                           -/


theorem conductor_eq_of_eq {y : S} (h : (R<x> : Set S) = R<y>) : conductor R x = conductor R y :=
  Ideal.ext fun _ => forall_congr' fun _ => Set.ext_iff.mp h _


theorem conductor_subset_adjoin : (conductor R x : Set S) ⊆ R<x> := fun y hy => by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x y : S
    hy : Membership.mem (↑(conductor R x)) y
    ⊢ Membership.mem (↑(Algebra.adjoin R (Singleton.singleton x))) y
  -/
  simpa only [mul_one] using hy 1
  /-
    🎉 no goals
  -/


theorem mem_conductor_iff {y : S} : y ∈ conductor R x ↔ ∀ b : S, y * b ∈ R<x> :=
  ⟨fun h => h, fun h => h⟩


theorem conductor_eq_top_of_adjoin_eq_top (h : R<x> = ⊤) : conductor R x = ⊤ := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : S
    h : Eq (Algebra.adjoin R (Singleton.singleton x)) Top.top
    ⊢ Eq (conductor R x) Top.top
  -/
  simp only [Ideal.eq_top_iff_one, mem_conductor_iff, h, mem_top, forall_const]
  /-
    🎉 no goals
  -/


theorem conductor_eq_top_of_powerBasis (pb : PowerBasis R S) : conductor R pb.gen = ⊤ :=
  conductor_eq_top_of_adjoin_eq_top pb.adjoin_gen_eq_top


open IsLocalization in
lemma mem_coeSubmodule_conductor {L} [CommRing L] [Algebra S L] [Algebra R L]
    [IsScalarTower R S L] [NoZeroSMulDivisors S L] {x : S} {y : L} :
    y ∈ coeSubmodule L (conductor R x) ↔ ∀ z : S,
      y * (algebraMap S L) z ∈ Algebra.adjoin R {algebraMap S L x} := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    L : Type u_3
    inst✝⁴ : CommRing L
    inst✝³ : Algebra S L
    inst✝² : Algebra R L
    inst✝¹ : IsScalarTower R S L
    inst✝ : NoZeroSMulDivisors S L
    x : S
    y : L
    ⊢ Iff (Membership.mem (IsLocalization.coeSubmodule L (conductor R x)) y) (∀ (z …
  -/
  cases subsingleton_or_nontrivial L
    /-
      case inl
      R : Type u_1
      S : Type u_2
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      inst✝⁵ : Algebra R S
      L : Type u_3
      inst✝⁴ : CommRing L
      inst✝³ : Algebra S L
      inst✝² : Algebra R L
      inst✝¹ : IsScalarTower R S L
      inst✝ : NoZeroSMulDivisors S L
      x : S
      y : L
      h✝ : Subsingleton L
      ⊢ Iff (Membership.mem (IsLocalization.coeSubmodule L (conductor R x)) y) (∀ (z …
    -/
  · rw [Subsingleton.elim (coeSubmodule L _) ⊤, Subsingleton.elim (Algebra.adjoin R _) ⊤]; simp
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/
  /-
    case inr
    R : Type u_1
    S : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    L : Type u_3
    inst✝⁴ : CommRing L
    inst✝³ : Algebra S L
    inst✝² : Algebra R L
    inst✝¹ : IsScalarTower R S L
    inst✝ : NoZeroSMulDivisors S L
    x : S
    y : L
    h✝ : Nontrivial L
    ⊢ Iff (Membership.mem (IsLocalization.coeSubmodule L (conductor R x)) y) (∀ (z …
  -/
  trans ∀ z, y * (algebraMap S L) z ∈ (Algebra.adjoin R {x}).map (IsScalarTower.toAlgHom R S L)
  · simp only [coeSubmodule, Submodule.mem_map, Algebra.linearMap_apply, Subalgebra.mem_map,
      IsScalarTower.coe_toAlgHom']
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      inst✝⁵ : Algebra R S
      L : Type u_3
      inst✝⁴ : CommRing L
      inst✝³ : Algebra S L
      inst✝² : Algebra R L
      inst✝¹ : IsScalarTower R S L
      inst✝ : NoZeroSMulDivisors S L
      x : S
      y : L
      h✝ : Nontrivial L
      ⊢ Iff (Exists fun y_1 => And (Membership.mem (conductor R x) y_1) (Eq ((algebr …
    -/
    constructor
      /-
        case mp
        R : Type u_1
        S : Type u_2
        inst✝⁷ : CommRing R
        inst✝⁶ : CommRing S
        inst✝⁵ : Algebra R S
        L : Type u_3
        inst✝⁴ : CommRing L
        inst✝³ : Algebra S L
        inst✝² : Algebra R L
        inst✝¹ : IsScalarTower R S L
        inst✝ : NoZeroSMulDivisors S L
        x : S
        y : L
        h✝ : Nontrivial L
        ⊢ (Exists fun y_1 => And (Membership.mem (conductor R x) y_1) (Eq ((algebraMap …
      -/
    · rintro ⟨y, hy, rfl⟩ z
      /-
        case mp.intro.intro
        R : Type u_1
        S : Type u_2
        inst✝⁷ : CommRing R
        inst✝⁶ : CommRing S
        inst✝⁵ : Algebra R S
        L : Type u_3
        inst✝⁴ : CommRing L
        inst✝³ : Algebra S L
        inst✝² : Algebra R L
        inst✝¹ : IsScalarTower R S L
        inst✝ : NoZeroSMulDivisors S L
        x : S
        h✝ : Nontrivial L
        y : S
        hy : Membership.mem (conductor R x) y
        z : S
        ⊢ Exists fun x_1 => And (Membership.mem (Algebra.adjoin R (Singleton.singleton …
      -/
      exact ⟨_, hy z, map_mul _ _ _⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr
        R : Type u_1
        S : Type u_2
        inst✝⁷ : CommRing R
        inst✝⁶ : CommRing S
        inst✝⁵ : Algebra R S
        L : Type u_3
        inst✝⁴ : CommRing L
        inst✝³ : Algebra S L
        inst✝² : Algebra R L
        inst✝¹ : IsScalarTower R S L
        inst✝ : NoZeroSMulDivisors S L
        x : S
        y : L
        h✝ : Nontrivial L
        ⊢ (∀ (z : S), Exists fun x_1 => And (Membership.mem (Algebra.adjoin R (Singlet …
      -/
    · intro H
      /-
        case mpr
        R : Type u_1
        S : Type u_2
        inst✝⁷ : CommRing R
        inst✝⁶ : CommRing S
        inst✝⁵ : Algebra R S
        L : Type u_3
        inst✝⁴ : CommRing L
        inst✝³ : Algebra S L
        inst✝² : Algebra R L
        inst✝¹ : IsScalarTower R S L
        inst✝ : NoZeroSMulDivisors S L
        x : S
        y : L
        h✝ : Nontrivial L
        H : ∀ (z : S), Exists fun x_1 => And (Membership.mem (Algebra.adjoin R (Single …
        ⊢ Exists fun y_1 => And (Membership.mem (conductor R x) y_1) (Eq ((algebraMap  …
      -/
      obtain ⟨y, _, e⟩ := H 1
      /-
        case mpr.intro.intro
        R : Type u_1
        S : Type u_2
        inst✝⁷ : CommRing R
        inst✝⁶ : CommRing S
        inst✝⁵ : Algebra R S
        L : Type u_3
        inst✝⁴ : CommRing L
        inst✝³ : Algebra S L
        inst✝² : Algebra R L
        inst✝¹ : IsScalarTower R S L
        inst✝ : NoZeroSMulDivisors S L
        x : S
        y✝ : L
        h✝ : Nontrivial L
        H : ∀ (z : S), Exists fun x_1 => And (Membership.mem (Algebra.adjoin R (Single …
        y : S
        left✝ : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) y
        e : Eq ((algebraMap S L) y) (HMul.hMul y✝ ((algebraMap S L) 1))
        ⊢ Exists fun y => And (Membership.mem (conductor R x) y) (Eq ((algebraMap S L) …
      -/
      rw [map_one, mul_one] at e
      /-
        case mpr.intro.intro
        R : Type u_1
        S : Type u_2
        inst✝⁷ : CommRing R
        inst✝⁶ : CommRing S
        inst✝⁵ : Algebra R S
        L : Type u_3
        inst✝⁴ : CommRing L
        inst✝³ : Algebra S L
        inst✝² : Algebra R L
        inst✝¹ : IsScalarTower R S L
        inst✝ : NoZeroSMulDivisors S L
        x : S
        y✝ : L
        h✝ : Nontrivial L
        H : ∀ (z : S), Exists fun x_1 => And (Membership.mem (Algebra.adjoin R (Single …
        y : S
        left✝ : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) y
        e : Eq ((algebraMap S L) y) y✝
        ⊢ Exists fun y => And (Membership.mem (conductor R x) y) (Eq ((algebraMap S L) …
      -/
      subst e
      simp only [← _root_.map_mul, (NoZeroSMulDivisors.algebraMap_injective S L).eq_iff,
        exists_eq_right] at H
      /-
        case mpr.intro.intro
        R : Type u_1
        S : Type u_2
        inst✝⁷ : CommRing R
        inst✝⁶ : CommRing S
        inst✝⁵ : Algebra R S
        L : Type u_3
        inst✝⁴ : CommRing L
        inst✝³ : Algebra S L
        inst✝² : Algebra R L
        inst✝¹ : IsScalarTower R S L
        inst✝ : NoZeroSMulDivisors S L
        x : S
        h✝ : Nontrivial L
        y : S
        left✝ : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) y
        H : ∀ (z : S), Membership.mem (Algebra.adjoin R (Singleton.singleton x)) (HMul …
        ⊢ Exists fun y_1 => And (Membership.mem (conductor R x) y_1) (Eq ((algebraMap  …
      -/
      exact ⟨_, H, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁷ : CommRing R
      inst✝⁶ : CommRing S
      inst✝⁵ : Algebra R S
      L : Type u_3
      inst✝⁴ : CommRing L
      inst✝³ : Algebra S L
      inst✝² : Algebra R L
      inst✝¹ : IsScalarTower R S L
      inst✝ : NoZeroSMulDivisors S L
      x : S
      y : L
      h✝ : Nontrivial L
      ⊢ Iff (∀ (z : S), Membership.mem (Subalgebra.map (IsScalarTower.toAlgHom R S L …
    -/
  · rw [AlgHom.map_adjoin, Set.image_singleton]; rfl
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- This technical lemma tell us that if `C` is the conductor of `R<x>` and `I` is an ideal of `R`
  then `p * (I * S) ⊆ I * R<x>` for any `p` in `C ∩ R` -/
theorem prod_mem_ideal_map_of_mem_conductor {p : R} {z : S}
    (hp : p ∈ Ideal.comap (algebraMap R S) (conductor R x)) (hz' : z ∈ I.map (algebraMap R S)) :
    algebraMap R S p * z ∈ algebraMap R<x> S '' ↑(I.map (algebraMap R R<x>)) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : S
    I : Ideal R
    p : R
    z : S
    hp : Membership.mem (Ideal.comap (algebraMap R S) (conductor R x)) p
    hz' : Membership.mem (Ideal.map (algebraMap R S) I) z
    ⊢ Membership.mem (Set.image ⇑(algebraMap (Subtype fun x_1 => Membership.mem (A …
  -/
  rw [Ideal.map, Ideal.span, Finsupp.mem_span_image_iff_linearCombination] at hz'
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : S
    I : Ideal R
    p : R
    z : S
    hp : Membership.mem (Ideal.comap (algebraMap R S) (conductor R x)) p
    hz' : Exists fun l => And (Membership.mem (Finsupp.supported S S ↑I) l) (Eq (( …
    ⊢ Membership.mem (Set.image ⇑(algebraMap (Subtype fun x_1 => Membership.mem (A …
  -/
  obtain ⟨l, H, H'⟩ := hz'
  /-
    case intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : S
    I : Ideal R
    p : R
    z : S
    hp : Membership.mem (Ideal.comap (algebraMap R S) (conductor R x)) p
    l : Finsupp R S
    H : Membership.mem (Finsupp.supported S S ↑I) l
    H' : Eq ((Finsupp.linearCombination S ⇑(algebraMap R S)) l) z
    ⊢ Membership.mem (Set.image ⇑(algebraMap (Subtype fun x_1 => Membership.mem (A …
  -/
  rw [Finsupp.linearCombination_apply] at H'
  /-
    case intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : S
    I : Ideal R
    p : R
    z : S
    hp : Membership.mem (Ideal.comap (algebraMap R S) (conductor R x)) p
    l : Finsupp R S
    H : Membership.mem (Finsupp.supported S S ↑I) l
    H' : Eq (l.sum fun i a => HSMul.hSMul a ((algebraMap R S) i)) z
    ⊢ Membership.mem (Set.image ⇑(algebraMap (Subtype fun x_1 => Membership.mem (A …
  -/
  rw [← H', mul_comm, Finsupp.sum_mul]
  have lem : ∀ {a : R}, a ∈ I → l a • algebraMap R S a * algebraMap R S p ∈
      algebraMap R<x> S '' I.map (algebraMap R R<x>) := by
    intro a ha
    rw [Algebra.id.smul_eq_mul, mul_assoc, mul_comm, mul_assoc, Set.mem_image]
    refine Exists.intro
        (algebraMap R R<x> a * ⟨l a * algebraMap R S p,
          show l a * algebraMap R S p ∈ R<x> from ?h⟩) ?_
    case h =>
      rw [mul_comm]
      exact mem_conductor_iff.mp (Ideal.mem_comap.mp hp) _
    · refine ⟨?_, ?_⟩
      · rw [mul_comm]
        apply Ideal.mul_mem_left (I.map (algebraMap R R<x>)) _ (Ideal.mem_map_of_mem _ ha)
      · simp only [RingHom.map_mul, mul_comm (algebraMap R S p) (l a)]
        rfl
  refine Finset.sum_induction _ (fun u => u ∈ algebraMap R<x> S '' I.map (algebraMap R R<x>))
      (fun a b => ?_) ?_ ?_
    /-
      case intro.intro.refine_1
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      p : R
      z : S
      hp : Membership.mem (Ideal.comap (algebraMap R S) (conductor R x)) p
      l : Finsupp R S
      H : Membership.mem (Finsupp.supported S S ↑I) l
      H' : Eq (l.sum fun i a => HSMul.hSMul a ((algebraMap R S) i)) z
      lem : ∀ {a : R}, Membership.mem I a → Membership.mem (Set.image ⇑(algebraMap ( …
      a b : S
      ⊢ (fun u => Membership.mem (Set.image ⇑(algebraMap (Subtype fun x_1 => Members …
    -/
  · rintro ⟨z, hz, rfl⟩ ⟨y, hy, rfl⟩
    /-
      case intro.intro.refine_1.intro.intro.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      p : R
      z✝ : S
      hp : Membership.mem (Ideal.comap (algebraMap R S) (conductor R x)) p
      l : Finsupp R S
      H : Membership.mem (Finsupp.supported S S ↑I) l
      H' : Eq (l.sum fun i a => HSMul.hSMul a ((algebraMap R S) i)) z✝
      lem : ∀ {a : R}, Membership.mem I a → Membership.mem (Set.image ⇑(algebraMap ( …
      z : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
      hz : Membership.mem (↑(Ideal.map (algebraMap R (Subtype fun x_1 => Membership. …
      y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
      hy : Membership.mem (↑(Ideal.map (algebraMap R (Subtype fun x_1 => Membership. …
      ⊢ Membership.mem (Set.image ⇑(algebraMap (Subtype fun x_1 => Membership.mem (A …
    -/
    rw [← RingHom.map_add]
    /-
      case intro.intro.refine_1.intro.intro.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      p : R
      z✝ : S
      hp : Membership.mem (Ideal.comap (algebraMap R S) (conductor R x)) p
      l : Finsupp R S
      H : Membership.mem (Finsupp.supported S S ↑I) l
      H' : Eq (l.sum fun i a => HSMul.hSMul a ((algebraMap R S) i)) z✝
      lem : ∀ {a : R}, Membership.mem I a → Membership.mem (Set.image ⇑(algebraMap ( …
      z : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
      hz : Membership.mem (↑(Ideal.map (algebraMap R (Subtype fun x_1 => Membership. …
      y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
      hy : Membership.mem (↑(Ideal.map (algebraMap R (Subtype fun x_1 => Membership. …
      ⊢ Membership.mem (Set.image ⇑(algebraMap (Subtype fun x_1 => Membership.mem (A …
    -/
    exact ⟨z + y, Ideal.add_mem _ (SetLike.mem_coe.mp hz) hy, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      p : R
      z : S
      hp : Membership.mem (Ideal.comap (algebraMap R S) (conductor R x)) p
      l : Finsupp R S
      H : Membership.mem (Finsupp.supported S S ↑I) l
      H' : Eq (l.sum fun i a => HSMul.hSMul a ((algebraMap R S) i)) z
      lem : ∀ {a : R}, Membership.mem I a → Membership.mem (Set.image ⇑(algebraMap ( …
      ⊢ (fun u => Membership.mem (Set.image ⇑(algebraMap (Subtype fun x_1 => Members …
    -/
  · exact ⟨0, SetLike.mem_coe.mpr <| Ideal.zero_mem _, RingHom.map_zero _⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_3
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      p : R
      z : S
      hp : Membership.mem (Ideal.comap (algebraMap R S) (conductor R x)) p
      l : Finsupp R S
      H : Membership.mem (Finsupp.supported S S ↑I) l
      H' : Eq (l.sum fun i a => HSMul.hSMul a ((algebraMap R S) i)) z
      lem : ∀ {a : R}, Membership.mem I a → Membership.mem (Set.image ⇑(algebraMap ( …
      ⊢ ∀ (x_1 : R), Membership.mem l.support x_1 → (fun u => Membership.mem (Set.im …
    -/
  · intro y hy
    /-
      case intro.intro.refine_3
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      p : R
      z : S
      hp : Membership.mem (Ideal.comap (algebraMap R S) (conductor R x)) p
      l : Finsupp R S
      H : Membership.mem (Finsupp.supported S S ↑I) l
      H' : Eq (l.sum fun i a => HSMul.hSMul a ((algebraMap R S) i)) z
      lem : ∀ {a : R}, Membership.mem I a → Membership.mem (Set.image ⇑(algebraMap ( …
      y : R
      hy : Membership.mem l.support y
      ⊢ Membership.mem (Set.image ⇑(algebraMap (Subtype fun x_1 => Membership.mem (A …
    -/
    exact lem ((Finsupp.mem_supported _ l).mp H hy)
    /-
      🎉 no goals
    -/


/-- A technical result telling us that `(I * S) ∩ R<x> = I * R<x>` for any ideal `I` of `R`. -/
theorem comap_map_eq_map_adjoin_of_coprime_conductor
    (hx : (conductor R x).comap (algebraMap R S) ⊔ I = ⊤)
    (h_alg : Function.Injective (algebraMap R<x> S)) :
    (I.map (algebraMap R S)).comap (algebraMap R<x> S) = I.map (algebraMap R R<x>) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : S
    I : Ideal R
    hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
    h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
    ⊢ Eq (Ideal.comap (algebraMap (Subtype fun x_1 => Membership.mem (Algebra.adjo …
  -/
  apply le_antisymm
  · -- This is adapted from [Neukirch1992]. Let `C = (conductor R x)`. The idea of the proof
    -- is that since `I` and `C ∩ R` are coprime, we have
    -- `(I * S) ∩ R<x> ⊆ (I + C) * ((I * S) ∩ R<x>) ⊆ I * R<x> + I * C * S ⊆ I * R<x>`.
    /-
      case a
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
      ⊢ LE.le (Ideal.comap (algebraMap (Subtype fun x_1 => Membership.mem (Algebra.a …
    -/
    intro y hy
    /-
      case a
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
      y : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton x …
      hy : Membership.mem (Ideal.comap (algebraMap (Subtype fun x_1 => Membership.me …
      ⊢ Membership.mem (Ideal.map (algebraMap R (Subtype fun x_1 => Membership.mem ( …
    -/
    obtain ⟨z, hz⟩ := y
    /-
      case a.mk
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
      z : S
      hz : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) z
      hy : Membership.mem (Ideal.comap (algebraMap (Subtype fun x_1 => Membership.me …
      ⊢ Membership.mem (Ideal.map (algebraMap R (Subtype fun x_1 => Membership.mem ( …
    -/
    obtain ⟨p, hp, q, hq, hpq⟩ := Submodule.mem_sup.mp ((Ideal.eq_top_iff_one _).mp hx)
    have temp : algebraMap R S p * z + algebraMap R S q * z = z := by
      simp only [← add_mul, ← RingHom.map_add (algebraMap R S), hpq, map_one, one_mul]
    suffices z ∈ algebraMap R<x> S '' I.map (algebraMap R R<x>) ↔
        (⟨z, hz⟩ : R<x>) ∈ I.map (algebraMap R R<x>) by
      rw [← this, ← temp]
      obtain ⟨a, ha⟩ := (Set.mem_image _ _ _).mp (prod_mem_ideal_map_of_mem_conductor hp
          (show z ∈ I.map (algebraMap R S) by rwa [Ideal.mem_comap] at hy))
      use a + algebraMap R R<x> q * ⟨z, hz⟩
      refine ⟨Ideal.add_mem (I.map (algebraMap R R<x>)) ha.left ?_, by
          simp only [ha.right, map_add, _root_.map_mul, add_right_inj]; rfl⟩
      rw [mul_comm]
      exact Ideal.mul_mem_left (I.map (algebraMap R R<x>)) _ (Ideal.mem_map_of_mem _ hq)
    refine ⟨fun h => ?_,
      fun h => (Set.mem_image _ _ _).mpr (Exists.intro ⟨z, hz⟩ ⟨by simp [h], rfl⟩)⟩
    /-
      case a.mk.intro.intro.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
      z : S
      hz : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) z
      hy : Membership.mem (Ideal.comap (algebraMap (Subtype fun x_1 => Membership.me …
      p : R
      hp : Membership.mem (Ideal.comap (algebraMap R S) (conductor R x)) p
      q : R
      hq : Membership.mem I q
      hpq : Eq (HAdd.hAdd p q) 1
      temp : Eq (HAdd.hAdd (HMul.hMul ((algebraMap R S) p) z) (HMul.hMul ((algebraMa …
      h : Membership.mem (Set.image ⇑(algebraMap (Subtype fun x_1 => Membership.mem  …
      ⊢ Membership.mem (Ideal.map (algebraMap R (Subtype fun x_1 => Membership.mem ( …
    -/
    obtain ⟨x₁, hx₁, hx₂⟩ := (Set.mem_image _ _ _).mp h
    have : x₁ = ⟨z, hz⟩ := by
      apply h_alg
      simp only [hx₂, algebraMap_eq_smul_one]
      rw [Submonoid.mk_smul, smul_eq_mul, mul_one]
    /-
      case a.mk.intro.intro.intro.intro.intro.intro
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
      z : S
      hz : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) z
      hy : Membership.mem (Ideal.comap (algebraMap (Subtype fun x_1 => Membership.me …
      p : R
      hp : Membership.mem (Ideal.comap (algebraMap R S) (conductor R x)) p
      q : R
      hq : Membership.mem I q
      hpq : Eq (HAdd.hAdd p q) 1
      temp : Eq (HAdd.hAdd (HMul.hMul ((algebraMap R S) p) z) (HMul.hMul ((algebraMa …
      h : Membership.mem (Set.image ⇑(algebraMap (Subtype fun x_1 => Membership.mem  …
      x₁ : Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Singleton.singleton  …
      hx₁ : Membership.mem (↑(Ideal.map (algebraMap R (Subtype fun x_1 => Membership …
      hx₂ : Eq ((algebraMap (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Si …
      this : Eq x₁ ⟨z, hz⟩
      ⊢ Membership.mem (Ideal.map (algebraMap R (Subtype fun x_1 => Membership.mem ( …
    -/
    rwa [← this]
    /-
      🎉 no goals
    -/
  · -- The converse inclusion is trivial
    /-
      case a
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
      ⊢ LE.le (Ideal.map (algebraMap R (Subtype fun x_1 => Membership.mem (Algebra.a …
    -/
    have : algebraMap R S = (algebraMap _ S).comp (algebraMap R R<x>) := by ext; rfl
    /-
      case a
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
      this : Eq (algebraMap R S) ((algebraMap (Subtype fun x_1 => Membership.mem (Al …
      ⊢ LE.le (Ideal.map (algebraMap R (Subtype fun x_1 => Membership.mem (Algebra.a …
    -/
    rw [this, ← Ideal.map_map]
    /-
      case a
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
      this : Eq (algebraMap R S) ((algebraMap (Subtype fun x_1 => Membership.mem (Al …
      ⊢ LE.le (Ideal.map (algebraMap R (Subtype fun x_1 => Membership.mem (Algebra.a …
    -/
    apply Ideal.le_comap_map
    /-
      🎉 no goals
    -/


/-- The canonical morphism of rings from `R<x> ⧸ (I*R<x>)` to `S ⧸ (I*S)` is an isomorphism
    when `I` and `(conductor R x) ∩ R` are coprime. -/
noncomputable def quotAdjoinEquivQuotMap (hx : (conductor R x).comap (algebraMap R S) ⊔ I = ⊤)
    (h_alg : Function.Injective (algebraMap R<x> S)) :
    R<x> ⧸ I.map (algebraMap R R<x>) ≃+* S ⧸ I.map (algebraMap R S) := by
  let f : R<x> ⧸ I.map (algebraMap R R<x>) →+* S ⧸ I.map (algebraMap R S) :=
    (Ideal.Quotient.lift (I.map (algebraMap R R<x>))
      ((Ideal.Quotient.mk (I.map (algebraMap R S))).comp (algebraMap R<x> S)) (fun r hr => by
      have : algebraMap R S = (algebraMap R<x> S).comp (algebraMap R R<x>) := by ext; rfl
      rw [RingHom.comp_apply, Ideal.Quotient.eq_zero_iff_mem, this, ← Ideal.map_map]
      exact Ideal.mem_map_of_mem _ hr))
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    x : S
    I : Ideal R
    hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
    h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
    f : RingHom (HasQuotient.Quotient (Subtype fun x_1 => Membership.mem (Algebra. …
    ⊢ RingEquiv (HasQuotient.Quotient (Subtype fun x_1 => Membership.mem (Algebra. …
  -/
  refine RingEquiv.ofBijective f ⟨?_, ?_⟩
  · --the kernel of the map is clearly `(I * S) ∩ R<x>`. To get injectivity, we need to show that
    --this is contained in `I * R<x>`, which is the content of the previous lemma.
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
      f : RingHom (HasQuotient.Quotient (Subtype fun x_1 => Membership.mem (Algebra. …
      ⊢ Function.Injective ⇑f
    -/
    refine RingHom.lift_injective_of_ker_le_ideal _ _ fun u hu => ?_
    rwa [RingHom.mem_ker, RingHom.comp_apply, Ideal.Quotient.eq_zero_iff_mem, ← Ideal.mem_comap,
      comap_map_eq_map_adjoin_of_coprime_conductor hx h_alg] at hu
  · -- Surjectivity follows from the surjectivity of the canonical map `R<x> → S ⧸ (I * S)`,
    -- which in turn follows from the fact that `I * S + (conductor R x) = S`.
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
      f : RingHom (HasQuotient.Quotient (Subtype fun x_1 => Membership.mem (Algebra. …
      ⊢ Function.Surjective ⇑f
    -/
    refine Ideal.Quotient.lift_surjective_of_surjective _ _ fun y => ?_
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
      f : RingHom (HasQuotient.Quotient (Subtype fun x_1 => Membership.mem (Algebra. …
      y : HasQuotient.Quotient S (Ideal.map (algebraMap R S) I)
      ⊢ Exists fun a => Eq (((Ideal.Quotient.mk (Ideal.map (algebraMap R S) I)).comp …
    -/
    obtain ⟨z, hz⟩ := Ideal.Quotient.mk_surjective y
    have : z ∈ conductor R x ⊔ I.map (algebraMap R S) := by
      suffices conductor R x ⊔ I.map (algebraMap R S) = ⊤ by simp only [this, Submodule.mem_top]
      rw [Ideal.eq_top_iff_one] at hx ⊢
      replace hx := Ideal.mem_map_of_mem (algebraMap R S) hx
      rw [Ideal.map_sup, RingHom.map_one] at hx
      exact (sup_le_sup
        (show ((conductor R x).comap (algebraMap R S)).map (algebraMap R S) ≤ conductor R x
          from Ideal.map_comap_le)
          (le_refl (I.map (algebraMap R S)))) hx
    /-
      case refine_2.intro
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      x : S
      I : Ideal R
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
      f : RingHom (HasQuotient.Quotient (Subtype fun x_1 => Membership.mem (Algebra. …
      y : HasQuotient.Quotient S (Ideal.map (algebraMap R S) I)
      z : S
      hz : Eq ((Ideal.Quotient.mk (Ideal.map (algebraMap R S) I)) z) y
      this : Membership.mem (Max.max (conductor R x) (Ideal.map (algebraMap R S) I)) z
      ⊢ Exists fun a => Eq (((Ideal.Quotient.mk (Ideal.map (algebraMap R S) I)).comp …
    -/
    rw [← Ideal.mem_quotient_iff_mem_sup, hz, Ideal.mem_map_iff_of_surjective] at this
      /-
        case refine_2.intro
        R : Type u_1
        S : Type u_2
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
        h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
        f : RingHom (HasQuotient.Quotient (Subtype fun x_1 => Membership.mem (Algebra. …
        y : HasQuotient.Quotient S (Ideal.map (algebraMap R S) I)
        z : S
        hz : Eq ((Ideal.Quotient.mk (Ideal.map (algebraMap R S) I)) z) y
        this : Exists fun x_1 => And (Membership.mem (conductor R x) x_1) (Eq ((Ideal. …
        ⊢ Exists fun a => Eq (((Ideal.Quotient.mk (Ideal.map (algebraMap R S) I)).comp …
      -/
    · obtain ⟨u, hu, hu'⟩ := this
      /-
        case refine_2.intro.intro.intro
        R : Type u_1
        S : Type u_2
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
        h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
        f : RingHom (HasQuotient.Quotient (Subtype fun x_1 => Membership.mem (Algebra. …
        y : HasQuotient.Quotient S (Ideal.map (algebraMap R S) I)
        z : S
        hz : Eq ((Ideal.Quotient.mk (Ideal.map (algebraMap R S) I)) z) y
        u : S
        hu : Membership.mem (conductor R x) u
        hu' : Eq ((Ideal.Quotient.mk (Ideal.map (algebraMap R S) I)) u) y
        ⊢ Exists fun a => Eq (((Ideal.Quotient.mk (Ideal.map (algebraMap R S) I)).comp …
      -/
      use ⟨u, conductor_subset_adjoin hu⟩
      /-
        case h
        R : Type u_1
        S : Type u_2
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
        h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
        f : RingHom (HasQuotient.Quotient (Subtype fun x_1 => Membership.mem (Algebra. …
        y : HasQuotient.Quotient S (Ideal.map (algebraMap R S) I)
        z : S
        hz : Eq ((Ideal.Quotient.mk (Ideal.map (algebraMap R S) I)) z) y
        u : S
        hu : Membership.mem (conductor R x) u
        hu' : Eq ((Ideal.Quotient.mk (Ideal.map (algebraMap R S) I)) u) y
        ⊢ Eq (((Ideal.Quotient.mk (Ideal.map (algebraMap R S) I)).comp (algebraMap (Su …
      -/
      simp only [← hu']
      /-
        case h
        R : Type u_1
        S : Type u_2
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
        h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
        f : RingHom (HasQuotient.Quotient (Subtype fun x_1 => Membership.mem (Algebra. …
        y : HasQuotient.Quotient S (Ideal.map (algebraMap R S) I)
        z : S
        hz : Eq ((Ideal.Quotient.mk (Ideal.map (algebraMap R S) I)) z) y
        u : S
        hu : Membership.mem (conductor R x) u
        hu' : Eq ((Ideal.Quotient.mk (Ideal.map (algebraMap R S) I)) u) y
        ⊢ Eq (((Ideal.Quotient.mk (Ideal.map (algebraMap R S) I)).comp (algebraMap (Su …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.hf
        R : Type u_1
        S : Type u_2
        inst✝² : CommRing R
        inst✝¹ : CommRing S
        inst✝ : Algebra R S
        x : S
        I : Ideal R
        hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
        h_alg : Function.Injective ⇑(algebraMap (Subtype fun x_1 => Membership.mem (Al …
        f : RingHom (HasQuotient.Quotient (Subtype fun x_1 => Membership.mem (Algebra. …
        y : HasQuotient.Quotient S (Ideal.map (algebraMap R S) I)
        z : S
        hz : Eq ((Ideal.Quotient.mk (Ideal.map (algebraMap R S) I)) z) y
        this : Membership.mem (Ideal.map (Ideal.Quotient.mk (Ideal.map (algebraMap R S …
        ⊢ Function.Surjective ⇑(Ideal.Quotient.mk (Ideal.map (algebraMap R S) I))
      -/
    · exact Ideal.Quotient.mk_surjective
      /-
        🎉 no goals
      -/


@[simp]
theorem quotAdjoinEquivQuotMap_apply_mk (hx : (conductor R x).comap (algebraMap R S) ⊔ I = ⊤)
    (h_alg : Function.Injective (algebraMap R<x> S)) (a : R<x>) :
    quotAdjoinEquivQuotMap hx h_alg (Ideal.Quotient.mk (I.map (algebraMap R R<x>)) a) =
      Ideal.Quotient.mk (I.map (algebraMap R S)) ↑a := rfl


open Classical in
/-- The first half of the **Kummer-Dedekind Theorem** in the monogenic case, stating that the prime
    factors of `I*S` are in bijection with those of the minimal polynomial of the generator of `S`
    over `R`, taken `mod I`. -/
noncomputable def normalizedFactorsMapEquivNormalizedFactorsMinPolyMk (hI : IsMaximal I)
    (hI' : I ≠ ⊥) (hx : (conductor R x).comap (algebraMap R S) ⊔ I = ⊤) (hx' : IsIntegral R x) :
    {J : Ideal S | J ∈ normalizedFactors (I.map (algebraMap R S))} ≃
      {d : (R ⧸ I)[X] |
        d ∈ normalizedFactors (Polynomial.map (Ideal.Quotient.mk I) (minpoly R x))} := by
  -- Porting note: Lean needs to be reminded about this so it does not time out
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    x : S
    I : Ideal R
    inst✝³ : IsDomain R
    inst✝² : IsIntegrallyClosed R
    inst✝¹ : IsDedekindDomain S
    inst✝ : NoZeroSMulDivisors R S
    hI : I.IsMaximal
    hI' : Ne I Bot.bot
    hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
    hx' : IsIntegral R x
    ⊢ Equiv ↑(setOf fun J => Membership.mem (UniqueFactorizationMonoid.normalizedF …
  -/
  have : IsPrincipalIdealRing (R ⧸ I)[X] := inferInstance
  let f : S ⧸ map (algebraMap R S) I ≃+*
    (R ⧸ I)[X] ⧸ span {Polynomial.map (Ideal.Quotient.mk I) (minpoly R x)} := by
    refine (quotAdjoinEquivQuotMap hx ?_).symm.trans
      (((Algebra.adjoin.powerBasis'
        hx').quotientEquivQuotientMinpolyMap I).toRingEquiv.trans (quotEquivOfEq ?_))
    · exact NoZeroSMulDivisors.algebraMap_injective (Algebra.adjoin R {x}) S
    · rw [Algebra.adjoin.powerBasis'_minpoly_gen hx']
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    x : S
    I : Ideal R
    inst✝³ : IsDomain R
    inst✝² : IsIntegrallyClosed R
    inst✝¹ : IsDedekindDomain S
    inst✝ : NoZeroSMulDivisors R S
    hI : I.IsMaximal
    hI' : Ne I Bot.bot
    hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
    hx' : IsIntegral R x
    this : IsPrincipalIdealRing (Polynomial (HasQuotient.Quotient R I))
    f : RingEquiv (HasQuotient.Quotient S (Ideal.map (algebraMap R S) I)) (HasQuot …
    ⊢ Equiv ↑(setOf fun J => Membership.mem (UniqueFactorizationMonoid.normalizedF …
  -/
  refine (normalizedFactorsEquivOfQuotEquiv f ?_ ?_).trans ?_
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      x : S
      I : Ideal R
      inst✝³ : IsDomain R
      inst✝² : IsIntegrallyClosed R
      inst✝¹ : IsDedekindDomain S
      inst✝ : NoZeroSMulDivisors R S
      hI : I.IsMaximal
      hI' : Ne I Bot.bot
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      hx' : IsIntegral R x
      this : IsPrincipalIdealRing (Polynomial (HasQuotient.Quotient R I))
      f : RingEquiv (HasQuotient.Quotient S (Ideal.map (algebraMap R S) I)) (HasQuot …
      ⊢ Ne (Ideal.map (algebraMap R S) I) Bot.bot
    -/
  · rwa [Ne, map_eq_bot_iff_of_injective (NoZeroSMulDivisors.algebraMap_injective R S), ← Ne]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      x : S
      I : Ideal R
      inst✝³ : IsDomain R
      inst✝² : IsIntegrallyClosed R
      inst✝¹ : IsDedekindDomain S
      inst✝ : NoZeroSMulDivisors R S
      hI : I.IsMaximal
      hI' : Ne I Bot.bot
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      hx' : IsIntegral R x
      this : IsPrincipalIdealRing (Polynomial (HasQuotient.Quotient R I))
      f : RingEquiv (HasQuotient.Quotient S (Ideal.map (algebraMap R S) I)) (HasQuot …
      ⊢ Ne (Ideal.span (Singleton.singleton (Polynomial.map (Ideal.Quotient.mk I) (m …
    -/
  · by_contra h
    exact (show Polynomial.map (Ideal.Quotient.mk I) (minpoly R x) ≠ 0 from
      Polynomial.map_monic_ne_zero (minpoly.monic hx')) (span_singleton_eq_bot.mp h)
    /-
      case refine_3
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      x : S
      I : Ideal R
      inst✝³ : IsDomain R
      inst✝² : IsIntegrallyClosed R
      inst✝¹ : IsDedekindDomain S
      inst✝ : NoZeroSMulDivisors R S
      hI : I.IsMaximal
      hI' : Ne I Bot.bot
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      hx' : IsIntegral R x
      this : IsPrincipalIdealRing (Polynomial (HasQuotient.Quotient R I))
      f : RingEquiv (HasQuotient.Quotient S (Ideal.map (algebraMap R S) I)) (HasQuot …
      ⊢ Equiv ↑(setOf fun M => Membership.mem (UniqueFactorizationMonoid.normalizedF …
    -/
  · refine (normalizedFactorsEquivSpanNormalizedFactors ?_).symm
    /-
      case refine_3
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      x : S
      I : Ideal R
      inst✝³ : IsDomain R
      inst✝² : IsIntegrallyClosed R
      inst✝¹ : IsDedekindDomain S
      inst✝ : NoZeroSMulDivisors R S
      hI : I.IsMaximal
      hI' : Ne I Bot.bot
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      hx' : IsIntegral R x
      this : IsPrincipalIdealRing (Polynomial (HasQuotient.Quotient R I))
      f : RingEquiv (HasQuotient.Quotient S (Ideal.map (algebraMap R S) I)) (HasQuot …
      ⊢ Ne (Polynomial.map (Ideal.Quotient.mk I) (minpoly R x)) 0
    -/
    exact Polynomial.map_monic_ne_zero (minpoly.monic hx')
    /-
      🎉 no goals
    -/


open Classical in
/-- The second half of the **Kummer-Dedekind Theorem** in the monogenic case, stating that the
    bijection `FactorsEquiv'` defined in the first half preserves multiplicities. -/
theorem emultiplicity_factors_map_eq_emultiplicity
    (hI : IsMaximal I) (hI' : I ≠ ⊥)
    (hx : (conductor R x).comap (algebraMap R S) ⊔ I = ⊤) (hx' : IsIntegral R x) {J : Ideal S}
    (hJ : J ∈ normalizedFactors (I.map (algebraMap R S))) :
    emultiplicity J (I.map (algebraMap R S)) =
      emultiplicity (↑(normalizedFactorsMapEquivNormalizedFactorsMinPolyMk hI hI' hx hx' ⟨J, hJ⟩))
        (Polynomial.map (Ideal.Quotient.mk I) (minpoly R x)) := by
  rw [normalizedFactorsMapEquivNormalizedFactorsMinPolyMk, Equiv.coe_trans, Function.comp_apply,
    emultiplicity_normalizedFactorsEquivSpanNormalizedFactors_symm_eq_emultiplicity,
    normalizedFactorsEquivOfQuotEquiv_emultiplicity_eq_emultiplicity]


open Classical in
/-- The **Kummer-Dedekind Theorem**. -/
theorem normalizedFactors_ideal_map_eq_normalizedFactors_min_poly_mk_map (hI : IsMaximal I)
    (hI' : I ≠ ⊥) (hx : (conductor R x).comap (algebraMap R S) ⊔ I = ⊤) (hx' : IsIntegral R x) :
    normalizedFactors (I.map (algebraMap R S)) =
      Multiset.map
        (fun f =>
          ((normalizedFactorsMapEquivNormalizedFactorsMinPolyMk hI hI' hx hx').symm f : Ideal S))
        (normalizedFactors (Polynomial.map (Ideal.Quotient.mk I) (minpoly R x))).attach := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    x : S
    I : Ideal R
    inst✝³ : IsDomain R
    inst✝² : IsIntegrallyClosed R
    inst✝¹ : IsDedekindDomain S
    inst✝ : NoZeroSMulDivisors R S
    hI : I.IsMaximal
    hI' : Ne I Bot.bot
    hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
    hx' : IsIntegral R x
    ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors (Ideal.map (algebraMap R S)  …
  -/
  ext J
  -- WLOG, assume J is a normalized factor
  /-
    case a
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    x : S
    I : Ideal R
    inst✝³ : IsDomain R
    inst✝² : IsIntegrallyClosed R
    inst✝¹ : IsDedekindDomain S
    inst✝ : NoZeroSMulDivisors R S
    hI : I.IsMaximal
    hI' : Ne I Bot.bot
    hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
    hx' : IsIntegral R x
    J : Ideal S
    ⊢ Eq (Multiset.count J (UniqueFactorizationMonoid.normalizedFactors (Ideal.map …
  -/
  by_cases hJ : J ∈ normalizedFactors (I.map (algebraMap R S))
  /-
    case pos
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    x : S
    I : Ideal R
    inst✝³ : IsDomain R
    inst✝² : IsIntegrallyClosed R
    inst✝¹ : IsDedekindDomain S
    inst✝ : NoZeroSMulDivisors R S
    hI : I.IsMaximal
    hI' : Ne I Bot.bot
    hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
    hx' : IsIntegral R x
    J : Ideal S
    hJ : Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Ideal.map (a …
    ⊢ Eq (Multiset.count J (UniqueFactorizationMonoid.normalizedFactors (Ideal.map …
  -/
  swap
    /-
      case neg
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      x : S
      I : Ideal R
      inst✝³ : IsDomain R
      inst✝² : IsIntegrallyClosed R
      inst✝¹ : IsDedekindDomain S
      inst✝ : NoZeroSMulDivisors R S
      hI : I.IsMaximal
      hI' : Ne I Bot.bot
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      hx' : IsIntegral R x
      J : Ideal S
      hJ : Not (Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Ideal.m …
      ⊢ Eq (Multiset.count J (UniqueFactorizationMonoid.normalizedFactors (Ideal.map …
    -/
  · rw [Multiset.count_eq_zero.mpr hJ, eq_comm, Multiset.count_eq_zero, Multiset.mem_map]
    /-
      case neg
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      x : S
      I : Ideal R
      inst✝³ : IsDomain R
      inst✝² : IsIntegrallyClosed R
      inst✝¹ : IsDedekindDomain S
      inst✝ : NoZeroSMulDivisors R S
      hI : I.IsMaximal
      hI' : Ne I Bot.bot
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      hx' : IsIntegral R x
      J : Ideal S
      hJ : Not (Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Ideal.m …
      ⊢ Not (Exists fun a => And (Membership.mem (UniqueFactorizationMonoid.normaliz …
    -/
    simp only [not_exists]
    /-
      case neg
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      x : S
      I : Ideal R
      inst✝³ : IsDomain R
      inst✝² : IsIntegrallyClosed R
      inst✝¹ : IsDedekindDomain S
      inst✝ : NoZeroSMulDivisors R S
      hI : I.IsMaximal
      hI' : Ne I Bot.bot
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      hx' : IsIntegral R x
      J : Ideal S
      hJ : Not (Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Ideal.m …
      ⊢ ∀ (x_1 : ↑(setOf fun d => Membership.mem (UniqueFactorizationMonoid.normaliz …
    -/
    rintro J' ⟨_, rfl⟩
    exact
      hJ ((normalizedFactorsMapEquivNormalizedFactorsMinPolyMk hI hI' hx hx').symm J').prop
  -- Then we just have to compare the multiplicities, which we already proved are equal.
  /-
    case pos
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    x : S
    I : Ideal R
    inst✝³ : IsDomain R
    inst✝² : IsIntegrallyClosed R
    inst✝¹ : IsDedekindDomain S
    inst✝ : NoZeroSMulDivisors R S
    hI : I.IsMaximal
    hI' : Ne I Bot.bot
    hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
    hx' : IsIntegral R x
    J : Ideal S
    hJ : Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Ideal.map (a …
    ⊢ Eq (Multiset.count J (UniqueFactorizationMonoid.normalizedFactors (Ideal.map …
  -/
  have := emultiplicity_factors_map_eq_emultiplicity hI hI' hx hx' hJ
  rw [emultiplicity_eq_count_normalizedFactors, emultiplicity_eq_count_normalizedFactors,
    UniqueFactorizationMonoid.normalize_normalized_factor _ hJ,
    UniqueFactorizationMonoid.normalize_normalized_factor, Nat.cast_inj] at this
    /-
      case pos
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      x : S
      I : Ideal R
      inst✝³ : IsDomain R
      inst✝² : IsIntegrallyClosed R
      inst✝¹ : IsDedekindDomain S
      inst✝ : NoZeroSMulDivisors R S
      hI : I.IsMaximal
      hI' : Ne I Bot.bot
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      hx' : IsIntegral R x
      J : Ideal S
      hJ : Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Ideal.map (a …
      this : Eq (Multiset.count J (UniqueFactorizationMonoid.normalizedFactors (Idea …
      ⊢ Eq (Multiset.count J (UniqueFactorizationMonoid.normalizedFactors (Ideal.map …
    -/
  · refine this.trans ?_
    -- Get rid of the `map` by applying the equiv to both sides.
    generalize hJ' :
      (normalizedFactorsMapEquivNormalizedFactorsMinPolyMk hI hI' hx hx') ⟨J, hJ⟩ = J'
    have : ((normalizedFactorsMapEquivNormalizedFactorsMinPolyMk hI hI' hx hx').symm J' : Ideal S) =
        J := by
      rw [← hJ', Equiv.symm_apply_apply _ _, Subtype.coe_mk]
    /-
      case pos
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      x : S
      I : Ideal R
      inst✝³ : IsDomain R
      inst✝² : IsIntegrallyClosed R
      inst✝¹ : IsDedekindDomain S
      inst✝ : NoZeroSMulDivisors R S
      hI : I.IsMaximal
      hI' : Ne I Bot.bot
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      hx' : IsIntegral R x
      J : Ideal S
      hJ : Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Ideal.map (a …
      this✝ : Eq (Multiset.count J (UniqueFactorizationMonoid.normalizedFactors (Ide …
      J' : ↑(setOf fun d => Membership.mem (UniqueFactorizationMonoid.normalizedFact …
      hJ' : Eq ((KummerDedekind.normalizedFactorsMapEquivNormalizedFactorsMinPolyMk  …
      this : Eq (↑((KummerDedekind.normalizedFactorsMapEquivNormalizedFactorsMinPoly …
      ⊢ Eq (Multiset.count (↑J') (UniqueFactorizationMonoid.normalizedFactors (Polyn …
    -/
    subst this
    -- Get rid of the `attach` by applying the subtype `coe` to both sides.
    rw [Multiset.count_map_eq_count' fun f =>
        ((normalizedFactorsMapEquivNormalizedFactorsMinPolyMk hI hI' hx hx').symm f :
          Ideal S),
      Multiset.count_attach]
      /-
        case pos.hf
        R : Type u_1
        S : Type u_2
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing S
        inst✝⁴ : Algebra R S
        x : S
        I : Ideal R
        inst✝³ : IsDomain R
        inst✝² : IsIntegrallyClosed R
        inst✝¹ : IsDedekindDomain S
        inst✝ : NoZeroSMulDivisors R S
        hI : I.IsMaximal
        hI' : Ne I Bot.bot
        hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
        hx' : IsIntegral R x
        J' : ↑(setOf fun d => Membership.mem (UniqueFactorizationMonoid.normalizedFact …
        hJ : Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Ideal.map (a …
        this : Eq (Multiset.count (↑((KummerDedekind.normalizedFactorsMapEquivNormaliz …
        hJ' : Eq ((KummerDedekind.normalizedFactorsMapEquivNormalizedFactorsMinPolyMk  …
        ⊢ Function.Injective fun f => ↑((KummerDedekind.normalizedFactorsMapEquivNorma …
      -/
    · exact Subtype.coe_injective.comp (Equiv.injective _)
      /-
        🎉 no goals
      -/
    /-
      case pos.a
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      x : S
      I : Ideal R
      inst✝³ : IsDomain R
      inst✝² : IsIntegrallyClosed R
      inst✝¹ : IsDedekindDomain S
      inst✝ : NoZeroSMulDivisors R S
      hI : I.IsMaximal
      hI' : Ne I Bot.bot
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      hx' : IsIntegral R x
      J : Ideal S
      hJ : Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Ideal.map (a …
      this : Eq ↑(Multiset.count J (UniqueFactorizationMonoid.normalizedFactors (Ide …
      ⊢ Membership.mem (UniqueFactorizationMonoid.normalizedFactors ?m.127270) ↑((Ku …
    -/
  · exact (normalizedFactorsMapEquivNormalizedFactorsMinPolyMk hI hI' hx hx' _).prop
    /-
      🎉 no goals
    -/
  · exact irreducible_of_normalized_factor _
        (normalizedFactorsMapEquivNormalizedFactorsMinPolyMk hI hI' hx hx' _).prop
    /-
      case pos.hb
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      x : S
      I : Ideal R
      inst✝³ : IsDomain R
      inst✝² : IsIntegrallyClosed R
      inst✝¹ : IsDedekindDomain S
      inst✝ : NoZeroSMulDivisors R S
      hI : I.IsMaximal
      hI' : Ne I Bot.bot
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      hx' : IsIntegral R x
      J : Ideal S
      hJ : Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Ideal.map (a …
      this : Eq (↑(Multiset.count (normalize J) (UniqueFactorizationMonoid.normalize …
      ⊢ Ne (Polynomial.map (Ideal.Quotient.mk I) (minpoly R x)) 0
    -/
  · exact Polynomial.map_monic_ne_zero (minpoly.monic hx')
    /-
      🎉 no goals
    -/
    /-
      case pos.ha
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      x : S
      I : Ideal R
      inst✝³ : IsDomain R
      inst✝² : IsIntegrallyClosed R
      inst✝¹ : IsDedekindDomain S
      inst✝ : NoZeroSMulDivisors R S
      hI : I.IsMaximal
      hI' : Ne I Bot.bot
      hx : Eq (Max.max (Ideal.comap (algebraMap R S) (conductor R x)) I) Top.top
      hx' : IsIntegral R x
      J : Ideal S
      hJ : Membership.mem (UniqueFactorizationMonoid.normalizedFactors (Ideal.map (a …
      this : Eq (emultiplicity J (Ideal.map (algebraMap R S) I)) (emultiplicity (↑(( …
      ⊢ Irreducible J
    -/
  · exact irreducible_of_normalized_factor _ hJ
    /-
      🎉 no goals
    -/
  · rwa [← bot_eq_zero, Ne,
      map_eq_bot_iff_of_injective (NoZeroSMulDivisors.algebraMap_injective R S)]


theorem Ideal.irreducible_map_of_irreducible_minpoly (hI : IsMaximal I) (hI' : I ≠ ⊥)
    (hx : (conductor R x).comap (algebraMap R S) ⊔ I = ⊤) (hx' : IsIntegral R x)
    (hf : Irreducible (Polynomial.map (Ideal.Quotient.mk I) (minpoly R x))) :
    Irreducible (I.map (algebraMap R S)) := by
  classical
  have mem_norm_factors : normalize (Polynomial.map (Ideal.Quotient.mk I) (minpoly R x)) ∈
      normalizedFactors (Polynomial.map (Ideal.Quotient.mk I) (minpoly R x)) := by
    simp [normalizedFactors_irreducible hf]
  suffices ∃ y, normalizedFactors (I.map (algebraMap R S)) = {y} by
    obtain ⟨y, hy⟩ := this
    have h := prod_normalizedFactors (show I.map (algebraMap R S) ≠ 0 by
          rwa [← bot_eq_zero, Ne,
            map_eq_bot_iff_of_injective (NoZeroSMulDivisors.algebraMap_injective R S)])
    rw [associated_iff_eq, hy, Multiset.prod_singleton] at h
    rw [← h]
    exact
      irreducible_of_normalized_factor y
        (show y ∈ normalizedFactors (I.map (algebraMap R S)) by simp [hy])
  rw [normalizedFactors_ideal_map_eq_normalizedFactors_min_poly_mk_map hI hI' hx hx']
  use ((normalizedFactorsMapEquivNormalizedFactorsMinPolyMk hI hI' hx hx').symm
        ⟨normalize (Polynomial.map (Ideal.Quotient.mk I) (minpoly R x)), mem_norm_factors⟩ :
      Ideal S)
  rw [Multiset.map_eq_singleton]
  use ⟨normalize (Polynomial.map (Ideal.Quotient.mk I) (minpoly R x)), mem_norm_factors⟩
  refine ⟨?_, rfl⟩
  apply Multiset.map_injective Subtype.coe_injective
  rw [Multiset.attach_map_val, Multiset.map_singleton, Subtype.coe_mk]
  exact normalizedFactors_irreducible hf


