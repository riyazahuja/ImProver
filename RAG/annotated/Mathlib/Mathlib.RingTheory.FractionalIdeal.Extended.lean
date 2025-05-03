/-- Given commutative rings `A` and `B` with respective localizations `IsLocalization M K` and
`IsLocalization N L`, and a ring homomorphism `f : A →+* B` satisfying `M ≤ Submonoid.comap f N`, a
fractional ideal `I` of `A` can be extended along `f` to a fractional ideal of `B`. -/
def extended (I : FractionalIdeal M K) : FractionalIdeal N L where
  val := span B <| (IsLocalization.map (S := K) L f hf) '' I
  property := by
    /-
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      I✝ J I : FractionalIdeal M K
      ⊢ IsFractional N (Submodule.span B (Set.image ⇑(IsLocalization.map L f hf) ↑I))
    -/
    have ⟨a, ha, frac⟩ := I.isFractional
    /-
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      I✝ J I : FractionalIdeal M K
      a : A
      ha : Membership.mem M a
      frac : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger A (HSMul.hS …
      ⊢ IsFractional N (Submodule.span B (Set.image ⇑(IsLocalization.map L f hf) ↑I))
    -/
    refine ⟨f a, hf ha, fun b hb ↦ ?_⟩
    refine span_induction (fun x hx ↦ ?_) ⟨0, by simp⟩
      (fun x y _ _ hx hy ↦ smul_add (f a) x y ▸ isInteger_add hx hy) (fun b c _ hc ↦ ?_) hb
      /-
        case refine_1
        A : Type u_1
        inst✝⁷ : CommRing A
        B : Type u_2
        inst✝⁶ : CommRing B
        f : RingHom A B
        K : Type u_3
        M : Submonoid A
        inst✝⁵ : CommRing K
        inst✝⁴ : Algebra A K
        inst✝³ : IsLocalization M K
        L : Type u_4
        N : Submonoid B
        inst✝² : CommRing L
        inst✝¹ : Algebra B L
        inst✝ : IsLocalization N L
        hf : LE.le M (Submonoid.comap f N)
        I✝ J I : FractionalIdeal M K
        a : A
        ha : Membership.mem M a
        frac : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger A (HSMul.hS …
        b : L
        hb : Membership.mem (Submodule.span B (Set.image ⇑(IsLocalization.map L f hf)  …
        x : L
        hx : Membership.mem (Set.image ⇑(IsLocalization.map L f hf) ↑I) x
        ⊢ IsLocalization.IsInteger B (HSMul.hSMul (f a) x)
      -/
    · rcases hx with ⟨k, kI, rfl⟩
      /-
        case refine_1.intro.intro
        A : Type u_1
        inst✝⁷ : CommRing A
        B : Type u_2
        inst✝⁶ : CommRing B
        f : RingHom A B
        K : Type u_3
        M : Submonoid A
        inst✝⁵ : CommRing K
        inst✝⁴ : Algebra A K
        inst✝³ : IsLocalization M K
        L : Type u_4
        N : Submonoid B
        inst✝² : CommRing L
        inst✝¹ : Algebra B L
        inst✝ : IsLocalization N L
        hf : LE.le M (Submonoid.comap f N)
        I✝ J I : FractionalIdeal M K
        a : A
        ha : Membership.mem M a
        frac : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger A (HSMul.hS …
        b : L
        hb : Membership.mem (Submodule.span B (Set.image ⇑(IsLocalization.map L f hf)  …
        k : K
        kI : Membership.mem (↑I) k
        ⊢ IsLocalization.IsInteger B (HSMul.hSMul (f a) ((IsLocalization.map L f hf) k))
      -/
      obtain ⟨c, hc⟩ := frac k kI
      /-
        case refine_1.intro.intro.intro
        A : Type u_1
        inst✝⁷ : CommRing A
        B : Type u_2
        inst✝⁶ : CommRing B
        f : RingHom A B
        K : Type u_3
        M : Submonoid A
        inst✝⁵ : CommRing K
        inst✝⁴ : Algebra A K
        inst✝³ : IsLocalization M K
        L : Type u_4
        N : Submonoid B
        inst✝² : CommRing L
        inst✝¹ : Algebra B L
        inst✝ : IsLocalization N L
        hf : LE.le M (Submonoid.comap f N)
        I✝ J I : FractionalIdeal M K
        a : A
        ha : Membership.mem M a
        frac : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger A (HSMul.hS …
        b : L
        hb : Membership.mem (Submodule.span B (Set.image ⇑(IsLocalization.map L f hf)  …
        k : K
        kI : Membership.mem (↑I) k
        c : A
        hc : Eq ((algebraMap A K) c) (HSMul.hSMul a k)
        ⊢ IsLocalization.IsInteger B (HSMul.hSMul (f a) ((IsLocalization.map L f hf) k))
      -/
      exact ⟨f c, by simp [← IsLocalization.map_smul, ← hc]⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        A : Type u_1
        inst✝⁷ : CommRing A
        B : Type u_2
        inst✝⁶ : CommRing B
        f : RingHom A B
        K : Type u_3
        M : Submonoid A
        inst✝⁵ : CommRing K
        inst✝⁴ : Algebra A K
        inst✝³ : IsLocalization M K
        L : Type u_4
        N : Submonoid B
        inst✝² : CommRing L
        inst✝¹ : Algebra B L
        inst✝ : IsLocalization N L
        hf : LE.le M (Submonoid.comap f N)
        I✝ J I : FractionalIdeal M K
        a : A
        ha : Membership.mem M a
        frac : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger A (HSMul.hS …
        b✝ : L
        hb : Membership.mem (Submodule.span B (Set.image ⇑(IsLocalization.map L f hf)  …
        b : B
        c : L
        x✝ : Membership.mem (Submodule.span B (Set.image ⇑(IsLocalization.map L f hf)  …
        hc : IsLocalization.IsInteger B (HSMul.hSMul (f a) c)
        ⊢ IsLocalization.IsInteger B (HSMul.hSMul (f a) (HSMul.hSMul b c))
      -/
    · rw [← smul_assoc, smul_eq_mul, mul_comm (f a), ← smul_eq_mul, smul_assoc]
      /-
        case refine_2
        A : Type u_1
        inst✝⁷ : CommRing A
        B : Type u_2
        inst✝⁶ : CommRing B
        f : RingHom A B
        K : Type u_3
        M : Submonoid A
        inst✝⁵ : CommRing K
        inst✝⁴ : Algebra A K
        inst✝³ : IsLocalization M K
        L : Type u_4
        N : Submonoid B
        inst✝² : CommRing L
        inst✝¹ : Algebra B L
        inst✝ : IsLocalization N L
        hf : LE.le M (Submonoid.comap f N)
        I✝ J I : FractionalIdeal M K
        a : A
        ha : Membership.mem M a
        frac : ∀ (b : K), Membership.mem (↑I) b → IsLocalization.IsInteger A (HSMul.hS …
        b✝ : L
        hb : Membership.mem (Submodule.span B (Set.image ⇑(IsLocalization.map L f hf)  …
        b : B
        c : L
        x✝ : Membership.mem (Submodule.span B (Set.image ⇑(IsLocalization.map L f hf)  …
        hc : IsLocalization.IsInteger B (HSMul.hSMul (f a) c)
        ⊢ IsLocalization.IsInteger B (HSMul.hSMul b (HSMul.hSMul (f a) c))
      -/
      exact isInteger_smul hc
      /-
        🎉 no goals
      -/


local notation "map_f" => (IsLocalization.map (S := K) L f hf)


lemma mem_extended_iff (x : L) : (x ∈ I.extended L hf) ↔ x ∈ span B (map_f '' I) := by
  /-
    A : Type u_1
    inst✝⁷ : CommRing A
    B : Type u_2
    inst✝⁶ : CommRing B
    f : RingHom A B
    K : Type u_3
    M : Submonoid A
    inst✝⁵ : CommRing K
    inst✝⁴ : Algebra A K
    inst✝³ : IsLocalization M K
    L : Type u_4
    N : Submonoid B
    inst✝² : CommRing L
    inst✝¹ : Algebra B L
    inst✝ : IsLocalization N L
    hf : LE.le M (Submonoid.comap f N)
    I : FractionalIdeal M K
    x : L
    ⊢ Iff (Membership.mem (FractionalIdeal.extended L hf I) x) (Membership.mem (Su …
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> { intro hx; simpa }
                  /-
                    🎉 no goals
                  -/


@[simp]
lemma coe_extended_eq_span : I.extended L hf = span B (map_f '' I) := by
  /-
    A : Type u_1
    inst✝⁷ : CommRing A
    B : Type u_2
    inst✝⁶ : CommRing B
    f : RingHom A B
    K : Type u_3
    M : Submonoid A
    inst✝⁵ : CommRing K
    inst✝⁴ : Algebra A K
    inst✝³ : IsLocalization M K
    L : Type u_4
    N : Submonoid B
    inst✝² : CommRing L
    inst✝¹ : Algebra B L
    inst✝ : IsLocalization N L
    hf : LE.le M (Submonoid.comap f N)
    I : FractionalIdeal M K
    ⊢ Eq (↑(FractionalIdeal.extended L hf I)) (Submodule.span B (Set.image ⇑(IsLoc …
  -/
  ext; simp [mem_coe, mem_extended_iff]
       /-
         🎉 no goals
       -/


@[simp]
theorem extended_zero : extended L hf (0 : FractionalIdeal M K) = 0 :=
                                                         /-
                                                           A : Type u_1
                                                           inst✝⁷ : CommRing A
                                                           B : Type u_2
                                                           inst✝⁶ : CommRing B
                                                           f : RingHom A B
                                                           K : Type u_3
                                                           M : Submonoid A
                                                           inst✝⁵ : CommRing K
                                                           inst✝⁴ : Algebra A K
                                                           inst✝³ : IsLocalization M K
                                                           L : Type u_4
                                                           N : Submonoid B
                                                           inst✝² : CommRing L
                                                           inst✝¹ : Algebra B L
                                                           inst✝ : IsLocalization N L
                                                           hf : LE.le M (Submonoid.comap f N)
                                                           ⊢ Eq (↑0) (Singleton.singleton 0)
                                                         -/
  have : ((0 : FractionalIdeal M K) : Set K) = {0} := by ext; simp
                                                              /-
                                                                🎉 no goals
                                                              -/
                               /-
                                 A : Type u_1
                                 inst✝⁷ : CommRing A
                                 B : Type u_2
                                 inst✝⁶ : CommRing B
                                 f : RingHom A B
                                 K : Type u_3
                                 M : Submonoid A
                                 inst✝⁵ : CommRing K
                                 inst✝⁴ : Algebra A K
                                 inst✝³ : IsLocalization M K
                                 L : Type u_4
                                 N : Submonoid B
                                 inst✝² : CommRing L
                                 inst✝¹ : Algebra B L
                                 inst✝ : IsLocalization N L
                                 hf : LE.le M (Submonoid.comap f N)
                                 this : Eq (↑0) (Singleton.singleton 0)
                                 ⊢ Eq ((fun I => ↑I) (FractionalIdeal.extended L hf 0)) ((fun I => ↑I) 0)
                               -/
  coeToSubmodule_injective (by simp [this])
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem extended_one : extended L hf (1 : FractionalIdeal M K) = 1 := by
  refine coeToSubmodule_injective <| Submodule.ext fun x ↦ ⟨fun hx ↦ span_induction
    ?_ (zero_mem _) (fun y z _ _ hy hz ↦ add_mem hy hz) (fun b y _ hy ↦ smul_mem _ b hy) hx, ?_⟩
    /-
      case refine_1
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      x : L
      ⊢ Membership.mem ((fun I => ↑I) 1) x → Membership.mem ((fun I => ↑I) (Fraction …
    -/
  · rintro ⟨b, _, rfl⟩
    /-
      case refine_1.intro.intro
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      b : B
      left✝ : Membership.mem (↑Top.top) b
      ⊢ Membership.mem ((fun I => ↑I) (FractionalIdeal.extended L hf 1)) ((Algebra.l …
    -/
    rw [Algebra.linearMap_apply, Algebra.algebraMap_eq_smul_one]
    /-
      case refine_1.intro.intro
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      b : B
      left✝ : Membership.mem (↑Top.top) b
      ⊢ Membership.mem ((fun I => ↑I) (FractionalIdeal.extended L hf 1)) (HSMul.hSMu …
    -/
    exact smul_mem _ _ <| subset_span ⟨1, by simp [one_mem_one]⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      x : L
      hx : Membership.mem ((fun I => ↑I) (FractionalIdeal.extended L hf 1)) x
      ⊢ ∀ (x : L), Membership.mem (Set.image ⇑(IsLocalization.map L f hf) ↑1) x → Me …
    -/
  · rintro _ ⟨_, ⟨a, ha, rfl⟩, rfl⟩
    /-
      case refine_2.intro.intro.intro.intro
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      x : L
      hx : Membership.mem ((fun I => ↑I) (FractionalIdeal.extended L hf 1)) x
      a : A
      ha : Membership.mem (↑Top.top) a
      ⊢ Membership.mem ((fun I => ↑I) 1) ((IsLocalization.map L f hf) ((Algebra.line …
    -/
    exact ⟨f a, ha, by rw [Algebra.linearMap_apply, Algebra.linearMap_apply, map_eq]⟩
    /-
      🎉 no goals
    -/


theorem extended_add : (I + J).extended L hf = (I.extended L hf) + (J.extended L hf) := by
  /-
    A : Type u_1
    inst✝⁷ : CommRing A
    B : Type u_2
    inst✝⁶ : CommRing B
    f : RingHom A B
    K : Type u_3
    M : Submonoid A
    inst✝⁵ : CommRing K
    inst✝⁴ : Algebra A K
    inst✝³ : IsLocalization M K
    L : Type u_4
    N : Submonoid B
    inst✝² : CommRing L
    inst✝¹ : Algebra B L
    inst✝ : IsLocalization N L
    hf : LE.le M (Submonoid.comap f N)
    I J : FractionalIdeal M K
    ⊢ Eq (FractionalIdeal.extended L hf (HAdd.hAdd I J)) (HAdd.hAdd (FractionalIde …
  -/
  apply coeToSubmodule_injective
  /-
    case a
    A : Type u_1
    inst✝⁷ : CommRing A
    B : Type u_2
    inst✝⁶ : CommRing B
    f : RingHom A B
    K : Type u_3
    M : Submonoid A
    inst✝⁵ : CommRing K
    inst✝⁴ : Algebra A K
    inst✝³ : IsLocalization M K
    L : Type u_4
    N : Submonoid B
    inst✝² : CommRing L
    inst✝¹ : Algebra B L
    inst✝ : IsLocalization N L
    hf : LE.le M (Submonoid.comap f N)
    I J : FractionalIdeal M K
    ⊢ Eq ((fun I => ↑I) (FractionalIdeal.extended L hf (HAdd.hAdd I J))) ((fun I = …
  -/
  simp only [coe_extended_eq_span, coe_add, Submodule.add_eq_sup, ← span_union, ← Set.image_union]
  /-
    case a
    A : Type u_1
    inst✝⁷ : CommRing A
    B : Type u_2
    inst✝⁶ : CommRing B
    f : RingHom A B
    K : Type u_3
    M : Submonoid A
    inst✝⁵ : CommRing K
    inst✝⁴ : Algebra A K
    inst✝³ : IsLocalization M K
    L : Type u_4
    N : Submonoid B
    inst✝² : CommRing L
    inst✝¹ : Algebra B L
    inst✝ : IsLocalization N L
    hf : LE.le M (Submonoid.comap f N)
    I J : FractionalIdeal M K
    ⊢ Eq (Submodule.span B (Set.image ⇑(IsLocalization.map L f hf) ↑(HAdd.hAdd I J …
  -/
  apply Submodule.span_eq_span
    /-
      case a.hs
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      I J : FractionalIdeal M K
      ⊢ HasSubset.Subset (Set.image ⇑(IsLocalization.map L f hf) ↑(HAdd.hAdd I J)) ↑ …
    -/
  · rintro _ ⟨y, hy, rfl⟩
    /-
      case a.hs.intro.intro
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      I J : FractionalIdeal M K
      y : K
      hy : Membership.mem (↑(HAdd.hAdd I J)) y
      ⊢ Membership.mem (↑(Submodule.span B (Set.image (⇑(IsLocalization.map L f hf)) …
    -/
    obtain ⟨i, hi, j, hj, rfl⟩ := (mem_add I J y).mp <| SetLike.mem_coe.mp hy
    /-
      case a.hs.intro.intro.intro.intro.intro.intro
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      I J : FractionalIdeal M K
      i : K
      hi : Membership.mem I i
      j : K
      hj : Membership.mem J j
      hy : Membership.mem (↑(HAdd.hAdd I J)) (HAdd.hAdd i j)
      ⊢ Membership.mem (↑(Submodule.span B (Set.image (⇑(IsLocalization.map L f hf)) …
    -/
    rw [RingHom.map_add]
    exact add_mem (Submodule.subset_span ⟨i, Set.mem_union_left _ hi, by simp⟩)
      (Submodule.subset_span ⟨j, Set.mem_union_right _ hj, by simp⟩)
    /-
      case a.ht
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      I J : FractionalIdeal M K
      ⊢ HasSubset.Subset (Set.image (⇑(IsLocalization.map L f hf)) (Union.union ↑I ↑ …
    -/
  · rintro _ ⟨y, hy, rfl⟩
    /-
      case a.ht.intro.intro
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      I J : FractionalIdeal M K
      y : K
      hy : Membership.mem (Union.union ↑I ↑J) y
      ⊢ Membership.mem (↑(Submodule.span B (Set.image ⇑(IsLocalization.map L f hf) ↑ …
    -/
    suffices y ∈ I + J from SetLike.mem_coe.mpr <| Submodule.subset_span ⟨y, by simp [this]⟩
    exact hy.elim (fun h ↦ (mem_add I J y).mpr ⟨y, h, 0, zero_mem J, add_zero y⟩)
      (fun h ↦ (mem_add I J y).mpr ⟨0, zero_mem I, y, h, zero_add y⟩)


theorem extended_mul : (I * J).extended L hf = (I.extended L hf) * (J.extended L hf) := by
  /-
    A : Type u_1
    inst✝⁷ : CommRing A
    B : Type u_2
    inst✝⁶ : CommRing B
    f : RingHom A B
    K : Type u_3
    M : Submonoid A
    inst✝⁵ : CommRing K
    inst✝⁴ : Algebra A K
    inst✝³ : IsLocalization M K
    L : Type u_4
    N : Submonoid B
    inst✝² : CommRing L
    inst✝¹ : Algebra B L
    inst✝ : IsLocalization N L
    hf : LE.le M (Submonoid.comap f N)
    I J : FractionalIdeal M K
    ⊢ Eq (FractionalIdeal.extended L hf (HMul.hMul I J)) (HMul.hMul (FractionalIde …
  -/
  apply coeToSubmodule_injective
  /-
    case a
    A : Type u_1
    inst✝⁷ : CommRing A
    B : Type u_2
    inst✝⁶ : CommRing B
    f : RingHom A B
    K : Type u_3
    M : Submonoid A
    inst✝⁵ : CommRing K
    inst✝⁴ : Algebra A K
    inst✝³ : IsLocalization M K
    L : Type u_4
    N : Submonoid B
    inst✝² : CommRing L
    inst✝¹ : Algebra B L
    inst✝ : IsLocalization N L
    hf : LE.le M (Submonoid.comap f N)
    I J : FractionalIdeal M K
    ⊢ Eq ((fun I => ↑I) (FractionalIdeal.extended L hf (HMul.hMul I J))) ((fun I = …
  -/
  simp only [coe_extended_eq_span, coe_mul, span_mul_span]
  /-
    case a
    A : Type u_1
    inst✝⁷ : CommRing A
    B : Type u_2
    inst✝⁶ : CommRing B
    f : RingHom A B
    K : Type u_3
    M : Submonoid A
    inst✝⁵ : CommRing K
    inst✝⁴ : Algebra A K
    inst✝³ : IsLocalization M K
    L : Type u_4
    N : Submonoid B
    inst✝² : CommRing L
    inst✝¹ : Algebra B L
    inst✝ : IsLocalization N L
    hf : LE.le M (Submonoid.comap f N)
    I J : FractionalIdeal M K
    ⊢ Eq (Submodule.span B (Set.image ⇑(IsLocalization.map L f hf) ↑(HMul.hMul I J …
  -/
  refine Submodule.span_eq_span (fun _ h ↦ ?_) (fun _ h ↦ ?_)
    /-
      case a.refine_1
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      I J : FractionalIdeal M K
      x✝ : L
      h : Membership.mem (Set.image ⇑(IsLocalization.map L f hf) ↑(HMul.hMul I J)) x✝
      ⊢ Membership.mem (↑(Submodule.span B (HMul.hMul (Set.image ⇑(IsLocalization.ma …
    -/
  · rcases h with ⟨x, hx, rfl⟩
    /-
      case a.refine_1.intro.intro
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      I J : FractionalIdeal M K
      x : K
      hx : Membership.mem (↑(HMul.hMul I J)) x
      ⊢ Membership.mem (↑(Submodule.span B (HMul.hMul (Set.image ⇑(IsLocalization.ma …
    -/
    replace hx : x ∈ (I : Submodule A K) * (J : Submodule A K) := coe_mul I J ▸ hx
    /-
      case a.refine_1.intro.intro
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      I J : FractionalIdeal M K
      x : K
      hx : Membership.mem (HMul.hMul ↑I ↑J) x
      ⊢ Membership.mem (↑(Submodule.span B (HMul.hMul (Set.image ⇑(IsLocalization.ma …
    -/
    rw [Submodule.mul_eq_span_mul_set] at hx
    refine span_induction (fun y hy ↦ ?_) (by simp) (fun y z _ _ hy hz ↦ ?_)
      (fun a y _ hy ↦ ?_) hx
      /-
        case a.refine_1.intro.intro.refine_1
        A : Type u_1
        inst✝⁷ : CommRing A
        B : Type u_2
        inst✝⁶ : CommRing B
        f : RingHom A B
        K : Type u_3
        M : Submonoid A
        inst✝⁵ : CommRing K
        inst✝⁴ : Algebra A K
        inst✝³ : IsLocalization M K
        L : Type u_4
        N : Submonoid B
        inst✝² : CommRing L
        inst✝¹ : Algebra B L
        inst✝ : IsLocalization N L
        hf : LE.le M (Submonoid.comap f N)
        I J : FractionalIdeal M K
        x : K
        hx : Membership.mem (Submodule.span A (HMul.hMul ↑↑I ↑↑J)) x
        y : K
        hy : Membership.mem (HMul.hMul ↑↑I ↑↑J) y
        ⊢ Membership.mem (↑(Submodule.span B (HMul.hMul (Set.image ⇑(IsLocalization.ma …
      -/
    · rcases Set.mem_mul.mp hy with ⟨i, hi, j, hj, rfl⟩
      exact subset_span <| Set.mem_mul.mpr
        ⟨map_f i, ⟨i, hi, by simp [hi]⟩, map_f j, ⟨j, hj, by simp [hj]⟩, by simp⟩
      /-
        case a.refine_1.intro.intro.refine_2
        A : Type u_1
        inst✝⁷ : CommRing A
        B : Type u_2
        inst✝⁶ : CommRing B
        f : RingHom A B
        K : Type u_3
        M : Submonoid A
        inst✝⁵ : CommRing K
        inst✝⁴ : Algebra A K
        inst✝³ : IsLocalization M K
        L : Type u_4
        N : Submonoid B
        inst✝² : CommRing L
        inst✝¹ : Algebra B L
        inst✝ : IsLocalization N L
        hf : LE.le M (Submonoid.comap f N)
        I J : FractionalIdeal M K
        x : K
        hx : Membership.mem (Submodule.span A (HMul.hMul ↑↑I ↑↑J)) x
        y z : K
        x✝¹ : Membership.mem (Submodule.span A (HMul.hMul ↑↑I ↑↑J)) y
        x✝ : Membership.mem (Submodule.span A (HMul.hMul ↑↑I ↑↑J)) z
        hy : Membership.mem (↑(Submodule.span B (HMul.hMul (Set.image ⇑(IsLocalization …
        hz : Membership.mem (↑(Submodule.span B (HMul.hMul (Set.image ⇑(IsLocalization …
        ⊢ Membership.mem (↑(Submodule.span B (HMul.hMul (Set.image ⇑(IsLocalization.ma …
      -/
    · exact map_add map_f y z ▸ Submodule.add_mem _ hy hz
      /-
        🎉 no goals
      -/
      /-
        case a.refine_1.intro.intro.refine_3
        A : Type u_1
        inst✝⁷ : CommRing A
        B : Type u_2
        inst✝⁶ : CommRing B
        f : RingHom A B
        K : Type u_3
        M : Submonoid A
        inst✝⁵ : CommRing K
        inst✝⁴ : Algebra A K
        inst✝³ : IsLocalization M K
        L : Type u_4
        N : Submonoid B
        inst✝² : CommRing L
        inst✝¹ : Algebra B L
        inst✝ : IsLocalization N L
        hf : LE.le M (Submonoid.comap f N)
        I J : FractionalIdeal M K
        x : K
        hx : Membership.mem (Submodule.span A (HMul.hMul ↑↑I ↑↑J)) x
        a : A
        y : K
        x✝ : Membership.mem (Submodule.span A (HMul.hMul ↑↑I ↑↑J)) y
        hy : Membership.mem (↑(Submodule.span B (HMul.hMul (Set.image ⇑(IsLocalization …
        ⊢ Membership.mem (↑(Submodule.span B (HMul.hMul (Set.image ⇑(IsLocalization.ma …
      -/
    · rw [Algebra.smul_def, map_mul, map_eq, ← Algebra.smul_def]
      /-
        case a.refine_1.intro.intro.refine_3
        A : Type u_1
        inst✝⁷ : CommRing A
        B : Type u_2
        inst✝⁶ : CommRing B
        f : RingHom A B
        K : Type u_3
        M : Submonoid A
        inst✝⁵ : CommRing K
        inst✝⁴ : Algebra A K
        inst✝³ : IsLocalization M K
        L : Type u_4
        N : Submonoid B
        inst✝² : CommRing L
        inst✝¹ : Algebra B L
        inst✝ : IsLocalization N L
        hf : LE.le M (Submonoid.comap f N)
        I J : FractionalIdeal M K
        x : K
        hx : Membership.mem (Submodule.span A (HMul.hMul ↑↑I ↑↑J)) x
        a : A
        y : K
        x✝ : Membership.mem (Submodule.span A (HMul.hMul ↑↑I ↑↑J)) y
        hy : Membership.mem (↑(Submodule.span B (HMul.hMul (Set.image ⇑(IsLocalization …
        ⊢ Membership.mem (↑(Submodule.span B (HMul.hMul (Set.image ⇑(IsLocalization.ma …
      -/
      exact smul_mem _ (f a) hy
      /-
        🎉 no goals
      -/
    /-
      case a.refine_2
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      I J : FractionalIdeal M K
      x✝ : L
      h : Membership.mem (HMul.hMul (Set.image ⇑(IsLocalization.map L f hf) ↑I) (Set …
      ⊢ Membership.mem (↑(Submodule.span B (Set.image ⇑(IsLocalization.map L f hf) ↑ …
    -/
  · rcases Set.mem_mul.mp h with ⟨y, ⟨i, hi, rfl⟩, z, ⟨j, hj, rfl⟩, rfl⟩
    /-
      case a.refine_2.intro.intro.intro.intro.intro.intro.intro.intro
      A : Type u_1
      inst✝⁷ : CommRing A
      B : Type u_2
      inst✝⁶ : CommRing B
      f : RingHom A B
      K : Type u_3
      M : Submonoid A
      inst✝⁵ : CommRing K
      inst✝⁴ : Algebra A K
      inst✝³ : IsLocalization M K
      L : Type u_4
      N : Submonoid B
      inst✝² : CommRing L
      inst✝¹ : Algebra B L
      inst✝ : IsLocalization N L
      hf : LE.le M (Submonoid.comap f N)
      I J : FractionalIdeal M K
      i : K
      hi : Membership.mem (↑I) i
      j : K
      hj : Membership.mem (↑J) j
      h : Membership.mem (HMul.hMul (Set.image ⇑(IsLocalization.map L f hf) ↑I) (Set …
      ⊢ Membership.mem (↑(Submodule.span B (Set.image ⇑(IsLocalization.map L f hf) ↑ …
    -/
    exact Submodule.subset_span ⟨i * j, mul_mem_mul hi hj, by simp⟩
    /-
      🎉 no goals
    -/


