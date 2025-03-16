variable {S} in
lemma IsLocalizedModule.linearIndependent_lift {ι} {v : ι → N} (hf : LinearIndependent S v) :
    ∃ w : ι → M, LinearIndependent R w := by
  /-
    R : Type u
    S : Type u'
    M : Type v
    N : Type v'
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R M
    inst✝⁵ : Module R N
    inst✝⁴ : Algebra R S
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    p : Submonoid R
    inst✝¹ : IsLocalization p S
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    hp : LE.le p (nonZeroDivisors R)
    ι : Type u_1
    v : ι → N
    hf : LinearIndependent S v
    ⊢ Exists fun w => LinearIndependent R w
  -/
  choose sec hsec using IsLocalizedModule.surj p f
  /-
    R : Type u
    S : Type u'
    M : Type v
    N : Type v'
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R M
    inst✝⁵ : Module R N
    inst✝⁴ : Algebra R S
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    p : Submonoid R
    inst✝¹ : IsLocalization p S
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    hp : LE.le p (nonZeroDivisors R)
    ι : Type u_1
    v : ι → N
    hf : LinearIndependent S v
    sec : N → Prod M (Subtype fun x => Membership.mem p x)
    hsec : ∀ (y : N), Eq (HSMul.hSMul (sec y).2 y) (f (sec y).1)
    ⊢ Exists fun w => LinearIndependent R w
  -/
  use fun i ↦ (sec (v i)).1
  /-
    case h
    R : Type u
    S : Type u'
    M : Type v
    N : Type v'
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R M
    inst✝⁵ : Module R N
    inst✝⁴ : Algebra R S
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    p : Submonoid R
    inst✝¹ : IsLocalization p S
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    hp : LE.le p (nonZeroDivisors R)
    ι : Type u_1
    v : ι → N
    hf : LinearIndependent S v
    sec : N → Prod M (Subtype fun x => Membership.mem p x)
    hsec : ∀ (y : N), Eq (HSMul.hSMul (sec y).2 y) (f (sec y).1)
    ⊢ LinearIndependent R fun i => (sec (v i)).1
  -/
  rw [linearIndependent_iff'] at hf ⊢
  /-
    case h
    R : Type u
    S : Type u'
    M : Type v
    N : Type v'
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R M
    inst✝⁵ : Module R N
    inst✝⁴ : Algebra R S
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    p : Submonoid R
    inst✝¹ : IsLocalization p S
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    hp : LE.le p (nonZeroDivisors R)
    ι : Type u_1
    v : ι → N
    hf : ∀ (s : Finset ι) (g : ι → S), Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) …
    sec : N → Prod M (Subtype fun x => Membership.mem p x)
    hsec : ∀ (y : N), Eq (HSMul.hSMul (sec y).2 y) (f (sec y).1)
    ⊢ ∀ (s : Finset ι) (g : ι → R), Eq (s.sum fun i => HSMul.hSMul (g i) (sec (v i …
  -/
  intro t g hg i hit
  /-
    case h
    R : Type u
    S : Type u'
    M : Type v
    N : Type v'
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R M
    inst✝⁵ : Module R N
    inst✝⁴ : Algebra R S
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    p : Submonoid R
    inst✝¹ : IsLocalization p S
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    hp : LE.le p (nonZeroDivisors R)
    ι : Type u_1
    v : ι → N
    hf : ∀ (s : Finset ι) (g : ι → S), Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) …
    sec : N → Prod M (Subtype fun x => Membership.mem p x)
    hsec : ∀ (y : N), Eq (HSMul.hSMul (sec y).2 y) (f (sec y).1)
    t : Finset ι
    g : ι → R
    hg : Eq (t.sum fun i => HSMul.hSMul (g i) (sec (v i)).1) 0
    i : ι
    hit : Membership.mem t i
    ⊢ Eq (g i) 0
  -/
  apply hp (sec (v i)).2.prop
  /-
    case h.a
    R : Type u
    S : Type u'
    M : Type v
    N : Type v'
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R M
    inst✝⁵ : Module R N
    inst✝⁴ : Algebra R S
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    p : Submonoid R
    inst✝¹ : IsLocalization p S
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    hp : LE.le p (nonZeroDivisors R)
    ι : Type u_1
    v : ι → N
    hf : ∀ (s : Finset ι) (g : ι → S), Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) …
    sec : N → Prod M (Subtype fun x => Membership.mem p x)
    hsec : ∀ (y : N), Eq (HSMul.hSMul (sec y).2 y) (f (sec y).1)
    t : Finset ι
    g : ι → R
    hg : Eq (t.sum fun i => HSMul.hSMul (g i) (sec (v i)).1) 0
    i : ι
    hit : Membership.mem t i
    ⊢ Eq (HMul.hMul (g i) ↑(sec (v i)).2) 0
  -/
  apply IsLocalization.injective S hp
  /-
    case h.a.a
    R : Type u
    S : Type u'
    M : Type v
    N : Type v'
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R M
    inst✝⁵ : Module R N
    inst✝⁴ : Algebra R S
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    p : Submonoid R
    inst✝¹ : IsLocalization p S
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    hp : LE.le p (nonZeroDivisors R)
    ι : Type u_1
    v : ι → N
    hf : ∀ (s : Finset ι) (g : ι → S), Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) …
    sec : N → Prod M (Subtype fun x => Membership.mem p x)
    hsec : ∀ (y : N), Eq (HSMul.hSMul (sec y).2 y) (f (sec y).1)
    t : Finset ι
    g : ι → R
    hg : Eq (t.sum fun i => HSMul.hSMul (g i) (sec (v i)).1) 0
    i : ι
    hit : Membership.mem t i
    ⊢ Eq ((algebraMap R S) (HMul.hMul (g i) ↑(sec (v i)).2)) ((algebraMap R S) 0)
  -/
  rw [map_zero]
  /-
    case h.a.a
    R : Type u
    S : Type u'
    M : Type v
    N : Type v'
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R M
    inst✝⁵ : Module R N
    inst✝⁴ : Algebra R S
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    p : Submonoid R
    inst✝¹ : IsLocalization p S
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    hp : LE.le p (nonZeroDivisors R)
    ι : Type u_1
    v : ι → N
    hf : ∀ (s : Finset ι) (g : ι → S), Eq (s.sum fun i => HSMul.hSMul (g i) (v i)) …
    sec : N → Prod M (Subtype fun x => Membership.mem p x)
    hsec : ∀ (y : N), Eq (HSMul.hSMul (sec y).2 y) (f (sec y).1)
    t : Finset ι
    g : ι → R
    hg : Eq (t.sum fun i => HSMul.hSMul (g i) (sec (v i)).1) 0
    i : ι
    hit : Membership.mem t i
    ⊢ Eq ((algebraMap R S) (HMul.hMul (g i) ↑(sec (v i)).2)) 0
  -/
  refine hf t (fun i ↦ algebraMap R S (g i * (sec (v i)).2)) ?_ _ hit
  simp only [map_mul, mul_smul, algebraMap_smul, ← Submonoid.smul_def,
    hsec, ← map_smul, ← map_sum, hg, map_zero]


lemma IsLocalizedModule.lift_rank_eq :
    Cardinal.lift.{v} (Module.rank S N) = Cardinal.lift.{v'} (Module.rank R M) := by
  /-
    R : Type u
    S : Type u'
    M : Type v
    N : Type v'
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R M
    inst✝⁵ : Module R N
    inst✝⁴ : Algebra R S
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    p : Submonoid R
    inst✝¹ : IsLocalization p S
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    hp : LE.le p (nonZeroDivisors R)
    ⊢ Eq (Cardinal.lift.{v, v'} (Module.rank S N)) (Cardinal.lift.{v', v} (Module. …
  -/
  cases' subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u
      S : Type u'
      M : Type v
      N : Type v'
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      hp : LE.le p (nonZeroDivisors R)
      h✝ : Subsingleton R
      ⊢ Eq (Cardinal.lift.{v, v'} (Module.rank S N)) (Cardinal.lift.{v', v} (Module. …
    -/
  · have := (algebraMap R S).codomain_trivial; simp only [rank_subsingleton, lift_one]
                                               /-
                                                 🎉 no goals
                                               -/
  /-
    case inr
    R : Type u
    S : Type u'
    M : Type v
    N : Type v'
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R M
    inst✝⁵ : Module R N
    inst✝⁴ : Algebra R S
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    p : Submonoid R
    inst✝¹ : IsLocalization p S
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    hp : LE.le p (nonZeroDivisors R)
    h✝ : Nontrivial R
    ⊢ Eq (Cardinal.lift.{v, v'} (Module.rank S N)) (Cardinal.lift.{v', v} (Module. …
  -/
  have := (IsLocalization.injective S hp).nontrivial
  /-
    case inr
    R : Type u
    S : Type u'
    M : Type v
    N : Type v'
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R M
    inst✝⁵ : Module R N
    inst✝⁴ : Algebra R S
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    p : Submonoid R
    inst✝¹ : IsLocalization p S
    f : LinearMap (RingHom.id R) M N
    inst✝ : IsLocalizedModule p f
    hp : LE.le p (nonZeroDivisors R)
    h✝ : Nontrivial R
    this : Nontrivial S
    ⊢ Eq (Cardinal.lift.{v, v'} (Module.rank S N)) (Cardinal.lift.{v', v} (Module. …
  -/
  apply le_antisymm <;>
    /-
      case inr.a
      R : Type u
      S : Type u'
      M : Type v
      N : Type v'
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      hp : LE.le p (nonZeroDivisors R)
      h✝ : Nontrivial R
      this : Nontrivial S
      ⊢ LE.le (Cardinal.lift.{v, v'} (Module.rank S N)) (Cardinal.lift.{v', v} (Modu …
    -/
    rw [Module.rank_def, lift_iSup (bddAbove_range _)] <;>
    /-
      case inr.a
      R : Type u
      S : Type u'
      M : Type v
      N : Type v'
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      hp : LE.le p (nonZeroDivisors R)
      h✝ : Nontrivial R
      this : Nontrivial S
      ⊢ LE.le (iSup fun i => Cardinal.lift.{v, v'} (Cardinal.mk ↑↑i)) (Cardinal.lift …
    -/
    apply ciSup_le' <;>
    /-
      case inr.a.h
      R : Type u
      S : Type u'
      M : Type v
      N : Type v'
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      hp : LE.le p (nonZeroDivisors R)
      h✝ : Nontrivial R
      this : Nontrivial S
      ⊢ ∀ (i : Subtype fun s => LinearIndependent S Subtype.val), LE.le (Cardinal.li …
    -/
    intro ⟨s, hs⟩
    /-
      case inr.a.h
      R : Type u
      S : Type u'
      M : Type v
      N : Type v'
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      hp : LE.le p (nonZeroDivisors R)
      h✝ : Nontrivial R
      this : Nontrivial S
      s : Set N
      hs : LinearIndependent S Subtype.val
      ⊢ LE.le (Cardinal.lift.{v, v'} (Cardinal.mk ↑↑⟨s, hs⟩)) (Cardinal.lift.{v', v} …
    -/
  · exact (IsLocalizedModule.linearIndependent_lift p f hp hs).choose_spec.cardinal_lift_le_rank
    /-
      🎉 no goals
    -/
    /-
      case inr.a.h
      R : Type u
      S : Type u'
      M : Type v
      N : Type v'
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      hp : LE.le p (nonZeroDivisors R)
      h✝ : Nontrivial R
      this : Nontrivial S
      s : Set M
      hs : LinearIndependent R Subtype.val
      ⊢ LE.le (Cardinal.lift.{v', v} (Cardinal.mk ↑↑⟨s, hs⟩)) (Cardinal.lift.{v, v'} …
    -/
  · choose sec hsec using IsLocalization.surj p (S := S)
    /-
      case inr.a.h
      R : Type u
      S : Type u'
      M : Type v
      N : Type v'
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      hp : LE.le p (nonZeroDivisors R)
      h✝ : Nontrivial R
      this : Nontrivial S
      s : Set M
      hs : LinearIndependent R Subtype.val
      sec : S → Prod R (Subtype fun x => Membership.mem p x)
      hsec : ∀ (z : S), Eq (HMul.hMul z ((algebraMap R S) ↑(sec z).2)) ((algebraMap  …
      ⊢ LE.le (Cardinal.lift.{v', v} (Cardinal.mk ↑↑⟨s, hs⟩)) (Cardinal.lift.{v, v'} …
    -/
    refine LinearIndependent.cardinal_lift_le_rank (ι := s) (v := fun i ↦ f i) ?_
    /-
      case inr.a.h
      R : Type u
      S : Type u'
      M : Type v
      N : Type v'
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      hp : LE.le p (nonZeroDivisors R)
      h✝ : Nontrivial R
      this : Nontrivial S
      s : Set M
      hs : LinearIndependent R Subtype.val
      sec : S → Prod R (Subtype fun x => Membership.mem p x)
      hsec : ∀ (z : S), Eq (HMul.hMul z ((algebraMap R S) ↑(sec z).2)) ((algebraMap  …
      ⊢ LinearIndependent S fun i => f ↑i
    -/
    rw [linearIndependent_iff'] at hs ⊢
    /-
      case inr.a.h
      R : Type u
      S : Type u'
      M : Type v
      N : Type v'
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      hp : LE.le p (nonZeroDivisors R)
      h✝ : Nontrivial R
      this : Nontrivial S
      s : Set M
      hs : ∀ (s_1 : Finset (Subtype fun x => Membership.mem s x)) (g : (Subtype fun  …
      sec : S → Prod R (Subtype fun x => Membership.mem p x)
      hsec : ∀ (z : S), Eq (HMul.hMul z ((algebraMap R S) ↑(sec z).2)) ((algebraMap  …
      ⊢ ∀ (s_1 : Finset ↑s) (g : ↑s → S), Eq (s_1.sum fun i => HSMul.hSMul (g i) (f  …
    -/
    intro t g hg i hit
    /-
      case inr.a.h
      R : Type u
      S : Type u'
      M : Type v
      N : Type v'
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : AddCommGroup M
      inst✝⁷ : AddCommGroup N
      inst✝⁶ : Module R M
      inst✝⁵ : Module R N
      inst✝⁴ : Algebra R S
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      p : Submonoid R
      inst✝¹ : IsLocalization p S
      f : LinearMap (RingHom.id R) M N
      inst✝ : IsLocalizedModule p f
      hp : LE.le p (nonZeroDivisors R)
      h✝ : Nontrivial R
      this : Nontrivial S
      s : Set M
      hs : ∀ (s_1 : Finset (Subtype fun x => Membership.mem s x)) (g : (Subtype fun  …
      sec : S → Prod R (Subtype fun x => Membership.mem p x)
      hsec : ∀ (z : S), Eq (HMul.hMul z ((algebraMap R S) ↑(sec z).2)) ((algebraMap  …
      t : Finset ↑s
      g : ↑s → S
      hg : Eq (t.sum fun i => HSMul.hSMul (g i) (f ↑i)) 0
      i : ↑s
      hit : Membership.mem t i
      ⊢ Eq (g i) 0
    -/
    apply (IsLocalization.map_units S (sec (g i)).2).mul_left_injective
    classical
    let u := fun (i : s) ↦ (t.erase i).prod (fun j ↦ (sec (g j)).2)
    have : f (t.sum fun i ↦ u i • (sec (g i)).1 • i) = f 0 := by
      convert congr_arg (t.prod (fun j ↦ (sec (g j)).2) • ·) hg
      · simp only [map_sum, map_smul, Submonoid.smul_def, Finset.smul_sum]
        apply Finset.sum_congr rfl
        intro j hj
        simp only [u, ← @IsScalarTower.algebraMap_smul R S N, Submonoid.coe_finset_prod, map_prod]
        rw [← hsec, mul_comm (g j), mul_smul, ← mul_smul, Finset.prod_erase_mul (h := hj)]
      rw [map_zero, smul_zero]
    obtain ⟨c, hc⟩ := IsLocalizedModule.exists_of_eq (S := p) this
    simp_rw [smul_zero, Finset.smul_sum, ← mul_smul, Submonoid.smul_def, ← mul_smul, mul_comm] at hc
    simp only [hsec, zero_mul, map_eq_zero_iff (algebraMap R S) (IsLocalization.injective S hp)]
    apply hp (c * u i).prop
    exact hs t _ hc _ hit


lemma IsLocalizedModule.rank_eq {N : Type v} [AddCommGroup N]
    [Module R N] [Module S N] [IsScalarTower R S N] (f : M →ₗ[R] N) [IsLocalizedModule p f] :
                                            /-
                                              R : Type u
                                              S : Type u'
                                              M : Type v
                                              inst✝¹⁰ : CommRing R
                                              inst✝⁹ : CommRing S
                                              inst✝⁸ : AddCommGroup M
                                              inst✝⁷ : Module R M
                                              inst✝⁶ : Algebra R S
                                              p : Submonoid R
                                              inst✝⁵ : IsLocalization p S
                                              hp : LE.le p (nonZeroDivisors R)
                                              N : Type v
                                              inst✝⁴ : AddCommGroup N
                                              inst✝³ : Module R N
                                              inst✝² : Module S N
                                              inst✝¹ : IsScalarTower R S N
                                              f : LinearMap (RingHom.id R) M N
                                              inst✝ : IsLocalizedModule p f
                                              ⊢ Eq (Module.rank S N) (Module.rank R M)
                                            -/
    Module.rank S N = Module.rank R M := by simpa using IsLocalizedModule.lift_rank_eq S p f hp
                                            /-
                                              🎉 no goals
                                            -/


variable (R M) in
theorem exists_set_linearIndependent_of_isDomain [IsDomain R] :
    ∃ s : Set M, #s = Module.rank R M ∧ LinearIndependent (ι := s) R Subtype.val := by
  obtain ⟨w, hw⟩ :=
    IsLocalizedModule.linearIndependent_lift R⁰ (LocalizedModule.mkLinearMap R⁰ M) le_rfl
      (Module.Free.chooseBasis (FractionRing R) (LocalizedModule R⁰ M)).linearIndependent
  /-
    case intro
    R : Type u
    M : Type v
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsDomain R
    w : Module.Free.ChooseBasisIndex (FractionRing R) (LocalizedModule (nonZeroDiv …
    hw : LinearIndependent R w
    ⊢ Exists fun s => And (Eq (Cardinal.mk ↑s) (Module.rank R M)) (LinearIndepende …
  -/
  refine ⟨Set.range w, ?_, (linearIndependent_subtype_range hw.injective).mpr hw⟩
  /-
    case intro
    R : Type u
    M : Type v
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsDomain R
    w : Module.Free.ChooseBasisIndex (FractionRing R) (LocalizedModule (nonZeroDiv …
    hw : LinearIndependent R w
    ⊢ Eq (Cardinal.mk ↑(Set.range w)) (Module.rank R M)
  -/
  apply Cardinal.lift_injective.{max u v}
  rw [Cardinal.mk_range_eq_of_injective hw.injective, ← Module.Free.rank_eq_card_chooseBasisIndex,
  IsLocalizedModule.lift_rank_eq (FractionRing R) R⁰ (LocalizedModule.mkLinearMap R⁰ M) le_rfl]


/-- The **rank-nullity theorem** for commutative domains. Also see `rank_quotient_add_rank`. -/
theorem rank_quotient_add_rank_of_isDomain [IsDomain R] (M' : Submodule R M) :
    Module.rank R (M ⧸ M') + Module.rank R M' = Module.rank R M := by
  /-
    R : Type u
    M : Type v
    inst✝³ : CommRing R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsDomain R
    M' : Submodule R M
    ⊢ Eq (HAdd.hAdd (Module.rank R (HasQuotient.Quotient M M')) (Module.rank R (Su …
  -/
  apply lift_injective.{max u v}
  rw [lift_add, ← IsLocalizedModule.lift_rank_eq (FractionRing R) R⁰ (M'.toLocalized R⁰) le_rfl,
    ← IsLocalizedModule.lift_rank_eq (FractionRing R) R⁰ (LocalizedModule.mkLinearMap R⁰ M) le_rfl,
    ← IsLocalizedModule.lift_rank_eq (FractionRing R) R⁰ (M'.toLocalizedQuotient R⁰) le_rfl,
    ← lift_add, rank_quotient_add_rank_of_divisionRing]


universe w in
instance IsDomain.hasRankNullity [IsDomain R] : HasRankNullity.{w} R where
  rank_quotient_add_rank := rank_quotient_add_rank_of_isDomain
  exists_set_linearIndependent M := exists_set_linearIndependent_of_isDomain R M


/-- A domain that is not (left) Ore is of infinite rank.
See [cohn_1995] Proposition 1.3.6 -/
lemma aleph0_le_rank_of_isEmpty_oreSet (hS : IsEmpty (OreLocalization.OreSet R⁰)) :
    ℵ₀ ≤ Module.rank R R := by
  classical
  rw [← not_nonempty_iff, OreLocalization.nonempty_oreSet_iff_of_noZeroDivisors] at hS
  push_neg at hS
  obtain ⟨r, s, h⟩ := hS
  refine Cardinal.aleph0_le.mpr fun n ↦ ?_
  suffices LinearIndependent R (fun (i : Fin n) ↦ r * s ^ (i : ℕ)) by
    simpa using this.cardinal_lift_le_rank
  suffices ∀ (g : ℕ → R) (x), (∑ i ∈ Finset.range n, g i • (r * s ^ (i + x))) = 0 →
      ∀ i < n, g i = 0 by
    refine Fintype.linearIndependent_iff.mpr fun g hg i ↦ ?_
    simpa only [dif_pos i.prop] using this (fun i ↦ if h : i < n then g ⟨i, h⟩ else 0) 0
      (by simp [← Fin.sum_univ_eq_sum_range, ← hg]) i i.prop
  intro g x hg i hin
  induction' n with n IH generalizing g x i
  · exact (hin.not_le (zero_le i)).elim
  · rw [Finset.sum_range_succ'] at hg
    by_cases hg0 : g 0 = 0
    · simp only [hg0, zero_smul, add_zero, add_assoc] at hg
      cases i; exacts [hg0, IH _ _ hg _ (Nat.succ_lt_succ_iff.mp hin)]
    simp only [MulOpposite.smul_eq_mul_unop, zero_add, ← add_comm _ x, pow_add _ _ x,
      ← mul_assoc, pow_succ, ← Finset.sum_mul, pow_zero, one_mul, smul_eq_mul] at hg
    rw [← neg_eq_iff_add_eq_zero, ← neg_mul, ← neg_mul] at hg
    have := mul_right_cancel₀ (mem_nonZeroDivisors_iff_ne_zero.mp (s ^ x).prop) hg
    exact (h _ ⟨(g 0), mem_nonZeroDivisors_iff_ne_zero.mpr (by simpa)⟩ this.symm).elim

-- TODO: Upgrade this to an iff. See [lam_1999] Exercise 10.21

lemma nonempty_oreSet_of_strongRankCondition [StrongRankCondition R] :
    Nonempty (OreLocalization.OreSet R⁰) := by
  /-
    R : Type u_1
    inst✝² : Ring R
    inst✝¹ : IsDomain R
    inst✝ : StrongRankCondition R
    ⊢ Nonempty (OreLocalization.OreSet (nonZeroDivisors R))
  -/
  by_contra h
  /-
    R : Type u_1
    inst✝² : Ring R
    inst✝¹ : IsDomain R
    inst✝ : StrongRankCondition R
    h : Not (Nonempty (OreLocalization.OreSet (nonZeroDivisors R)))
    ⊢ False
  -/
  have := aleph0_le_rank_of_isEmpty_oreSet (not_nonempty_iff.mp h)
  /-
    R : Type u_1
    inst✝² : Ring R
    inst✝¹ : IsDomain R
    inst✝ : StrongRankCondition R
    h : Not (Nonempty (OreLocalization.OreSet (nonZeroDivisors R)))
    this : LE.le Cardinal.aleph0 (Module.rank R R)
    ⊢ False
  -/
  rw [rank_self] at this
  /-
    R : Type u_1
    inst✝² : Ring R
    inst✝¹ : IsDomain R
    inst✝ : StrongRankCondition R
    h : Not (Nonempty (OreLocalization.OreSet (nonZeroDivisors R)))
    this : LE.le Cardinal.aleph0 1
    ⊢ False
  -/
  exact this.not_lt one_lt_aleph0
  /-
    🎉 no goals
  -/


