/-- The submodule spanning the differences of a (possibly empty) set of points. -/
def vectorSpan (s : Set P) : Submodule k V :=
  Submodule.span k (s -ᵥ s)


/-- The definition of `vectorSpan`, for rewriting. -/
theorem vectorSpan_def (s : Set P) : vectorSpan k s = Submodule.span k (s -ᵥ s) :=
  rfl


/-- `vectorSpan` is monotone. -/
theorem vectorSpan_mono {s₁ s₂ : Set P} (h : s₁ ⊆ s₂) : vectorSpan k s₁ ≤ vectorSpan k s₂ :=
  Submodule.span_mono (vsub_self_mono h)


/-- The `vectorSpan` of the empty set is `⊥`. -/
@[simp]
theorem vectorSpan_empty : vectorSpan k (∅ : Set P) = (⊥ : Submodule k V) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ⊢ Eq (vectorSpan k EmptyCollection.emptyCollection) Bot.bot
  -/
  rw [vectorSpan_def, vsub_empty, Submodule.span_empty]
  /-
    🎉 no goals
  -/


/-- The `vectorSpan` of a single point is `⊥`. -/
@[simp]
                                                                            /-
                                                                              k : Type u_1
                                                                              V : Type u_2
                                                                              P : Type u_3
                                                                              inst✝³ : Ring k
                                                                              inst✝² : AddCommGroup V
                                                                              inst✝¹ : Module k V
                                                                              inst✝ : AddTorsor V P
                                                                              p : P
                                                                              ⊢ Eq (vectorSpan k (Singleton.singleton p)) Bot.bot
                                                                            -/
theorem vectorSpan_singleton (p : P) : vectorSpan k ({p} : Set P) = ⊥ := by simp [vectorSpan_def]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- The `s -ᵥ s` lies within the `vectorSpan k s`. -/
theorem vsub_set_subset_vectorSpan (s : Set P) : s -ᵥ s ⊆ ↑(vectorSpan k s) :=
  Submodule.subset_span


/-- Each pairwise difference is in the `vectorSpan`. -/
theorem vsub_mem_vectorSpan {s : Set P} {p1 p2 : P} (hp1 : p1 ∈ s) (hp2 : p2 ∈ s) :
    p1 -ᵥ p2 ∈ vectorSpan k s :=
  vsub_set_subset_vectorSpan k s (vsub_mem_vsub hp1 hp2)


/-- The points in the affine span of a (possibly empty) set of points. Use `affineSpan` instead to
get an `AffineSubspace k P`. -/
def spanPoints (s : Set P) : Set P :=
  { p | ∃ p1 ∈ s, ∃ v ∈ vectorSpan k s, p = v +ᵥ p1 }


/-- A point in a set is in its affine span. -/
theorem mem_spanPoints (p : P) (s : Set P) : p ∈ s → p ∈ spanPoints k s
  | hp => ⟨p, hp, 0, Submodule.zero_mem _, (zero_vadd V p).symm⟩


/-- A set is contained in its `spanPoints`. -/
theorem subset_spanPoints (s : Set P) : s ⊆ spanPoints k s := fun p => mem_spanPoints k p s


/-- The `spanPoints` of a set is nonempty if and only if that set is. -/
@[simp]
theorem spanPoints_nonempty (s : Set P) : (spanPoints k s).Nonempty ↔ s.Nonempty := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    ⊢ Iff (spanPoints k s).Nonempty s.Nonempty
  -/
  constructor
    /-
      case mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      ⊢ (spanPoints k s).Nonempty → s.Nonempty
    -/
  · contrapose
    /-
      case mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      ⊢ Not s.Nonempty → Not (spanPoints k s).Nonempty
    -/
    rw [Set.not_nonempty_iff_eq_empty, Set.not_nonempty_iff_eq_empty]
    /-
      case mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      ⊢ Eq s EmptyCollection.emptyCollection → Eq (spanPoints k s) EmptyCollection.e …
    -/
    intro h
    /-
      case mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      h : Eq s EmptyCollection.emptyCollection
      ⊢ Eq (spanPoints k s) EmptyCollection.emptyCollection
    -/
    simp [h, spanPoints]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      ⊢ s.Nonempty → (spanPoints k s).Nonempty
    -/
  · exact fun h => h.mono (subset_spanPoints _ _)
    /-
      🎉 no goals
    -/


/-- Adding a point in the affine span and a vector in the spanning submodule produces a point in the
affine span. -/
theorem vadd_mem_spanPoints_of_mem_spanPoints_of_mem_vectorSpan {s : Set P} {p : P} {v : V}
    (hp : p ∈ spanPoints k s) (hv : v ∈ vectorSpan k s) : v +ᵥ p ∈ spanPoints k s := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p : P
    v : V
    hp : Membership.mem (spanPoints k s) p
    hv : Membership.mem (vectorSpan k s) v
    ⊢ Membership.mem (spanPoints k s) (HVAdd.hVAdd v p)
  -/
  rcases hp with ⟨p2, ⟨hp2, ⟨v2, ⟨hv2, hv2p⟩⟩⟩⟩
  /-
    case intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p : P
    v : V
    hv : Membership.mem (vectorSpan k s) v
    p2 : P
    hp2 : Membership.mem s p2
    v2 : V
    hv2 : Membership.mem (vectorSpan k s) v2
    hv2p : Eq p (HVAdd.hVAdd v2 p2)
    ⊢ Membership.mem (spanPoints k s) (HVAdd.hVAdd v p)
  -/
  rw [hv2p, vadd_vadd]
  /-
    case intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p : P
    v : V
    hv : Membership.mem (vectorSpan k s) v
    p2 : P
    hp2 : Membership.mem s p2
    v2 : V
    hv2 : Membership.mem (vectorSpan k s) v2
    hv2p : Eq p (HVAdd.hVAdd v2 p2)
    ⊢ Membership.mem (spanPoints k s) (HVAdd.hVAdd (HAdd.hAdd v v2) p2)
  -/
  exact ⟨p2, hp2, v + v2, (vectorSpan k s).add_mem hv hv2, rfl⟩
  /-
    🎉 no goals
  -/


/-- Subtracting two points in the affine span produces a vector in the spanning submodule. -/
theorem vsub_mem_vectorSpan_of_mem_spanPoints_of_mem_spanPoints {s : Set P} {p1 p2 : P}
    (hp1 : p1 ∈ spanPoints k s) (hp2 : p2 ∈ spanPoints k s) : p1 -ᵥ p2 ∈ vectorSpan k s := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p1 p2 : P
    hp1 : Membership.mem (spanPoints k s) p1
    hp2 : Membership.mem (spanPoints k s) p2
    ⊢ Membership.mem (vectorSpan k s) (VSub.vsub p1 p2)
  -/
  rcases hp1 with ⟨p1a, ⟨hp1a, ⟨v1, ⟨hv1, hv1p⟩⟩⟩⟩
  /-
    case intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p1 p2 : P
    hp2 : Membership.mem (spanPoints k s) p2
    p1a : P
    hp1a : Membership.mem s p1a
    v1 : V
    hv1 : Membership.mem (vectorSpan k s) v1
    hv1p : Eq p1 (HVAdd.hVAdd v1 p1a)
    ⊢ Membership.mem (vectorSpan k s) (VSub.vsub p1 p2)
  -/
  rcases hp2 with ⟨p2a, ⟨hp2a, ⟨v2, ⟨hv2, hv2p⟩⟩⟩⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p1 p2 p1a : P
    hp1a : Membership.mem s p1a
    v1 : V
    hv1 : Membership.mem (vectorSpan k s) v1
    hv1p : Eq p1 (HVAdd.hVAdd v1 p1a)
    p2a : P
    hp2a : Membership.mem s p2a
    v2 : V
    hv2 : Membership.mem (vectorSpan k s) v2
    hv2p : Eq p2 (HVAdd.hVAdd v2 p2a)
    ⊢ Membership.mem (vectorSpan k s) (VSub.vsub p1 p2)
  -/
  rw [hv1p, hv2p, vsub_vadd_eq_vsub_sub (v1 +ᵥ p1a), vadd_vsub_assoc, add_comm, add_sub_assoc]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p1 p2 p1a : P
    hp1a : Membership.mem s p1a
    v1 : V
    hv1 : Membership.mem (vectorSpan k s) v1
    hv1p : Eq p1 (HVAdd.hVAdd v1 p1a)
    p2a : P
    hp2a : Membership.mem s p2a
    v2 : V
    hv2 : Membership.mem (vectorSpan k s) v2
    hv2p : Eq p2 (HVAdd.hVAdd v2 p2a)
    ⊢ Membership.mem (vectorSpan k s) (HAdd.hAdd (VSub.vsub p1a p2a) (HSub.hSub v1 …
  -/
  have hv1v2 : v1 - v2 ∈ vectorSpan k s := (vectorSpan k s).sub_mem hv1 hv2
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p1 p2 p1a : P
    hp1a : Membership.mem s p1a
    v1 : V
    hv1 : Membership.mem (vectorSpan k s) v1
    hv1p : Eq p1 (HVAdd.hVAdd v1 p1a)
    p2a : P
    hp2a : Membership.mem s p2a
    v2 : V
    hv2 : Membership.mem (vectorSpan k s) v2
    hv2p : Eq p2 (HVAdd.hVAdd v2 p2a)
    hv1v2 : Membership.mem (vectorSpan k s) (HSub.hSub v1 v2)
    ⊢ Membership.mem (vectorSpan k s) (HAdd.hAdd (VSub.vsub p1a p2a) (HSub.hSub v1 …
  -/
  refine (vectorSpan k s).add_mem ?_ hv1v2
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p1 p2 p1a : P
    hp1a : Membership.mem s p1a
    v1 : V
    hv1 : Membership.mem (vectorSpan k s) v1
    hv1p : Eq p1 (HVAdd.hVAdd v1 p1a)
    p2a : P
    hp2a : Membership.mem s p2a
    v2 : V
    hv2 : Membership.mem (vectorSpan k s) v2
    hv2p : Eq p2 (HVAdd.hVAdd v2 p2a)
    hv1v2 : Membership.mem (vectorSpan k s) (HSub.hSub v1 v2)
    ⊢ Membership.mem (vectorSpan k s) (VSub.vsub p1a p2a)
  -/
  exact vsub_mem_vectorSpan k hp1a hp2a
  /-
    🎉 no goals
  -/


/-- An `AffineSubspace k P` is a subset of an `AffineSpace V P` that, if not empty, has an affine
space structure induced by a corresponding subspace of the `Module k V`. -/
structure AffineSubspace (k : Type*) {V : Type*} (P : Type*) [Ring k] [AddCommGroup V]
  [Module k V] [AffineSpace V P] where
  /-- The affine subspace seen as a subset. -/
  carrier : Set P
  smul_vsub_vadd_mem :
    ∀ (c : k) {p1 p2 p3 : P},
      p1 ∈ carrier → p2 ∈ carrier → p3 ∈ carrier → c • (p1 -ᵥ p2 : V) +ᵥ p3 ∈ carrier


/-- Reinterpret `p : Submodule k V` as an `AffineSubspace k V`. -/
def toAffineSubspace (p : Submodule k V) : AffineSubspace k V where
  carrier := p
  smul_vsub_vadd_mem _ _ _ _ h₁ h₂ h₃ := p.add_mem (p.smul_mem _ (p.sub_mem h₁ h₂)) h₃


instance : SetLike (AffineSubspace k P) P where
  coe := carrier
                             /-
                               k : Type u_1
                               V : Type u_2
                               P : Type u_3
                               inst✝³ : Ring k
                               inst✝² : AddCommGroup V
                               inst✝¹ : Module k V
                               inst✝ : AddTorsor V P
                               p q : AffineSubspace k P
                               x✝ : Eq p.carrier q.carrier
                               ⊢ Eq p q
                             -/
  coe_injective' p q _ := by cases p; cases q; congr
                                               /-
                                                 🎉 no goals
                                               -/


/-- A point is in an affine subspace coerced to a set if and only if it is in that affine
subspace. -/
-- Porting note: removed `simp`, proof is `simp only [SetLike.mem_coe]`
theorem mem_coe (p : P) (s : AffineSubspace k P) : p ∈ (s : Set P) ↔ p ∈ s :=
  Iff.rfl


/-- The direction of an affine subspace is the submodule spanned by
the pairwise differences of points.  (Except in the case of an empty
affine subspace, where the direction is the zero submodule, every
vector in the direction is the difference of two points in the affine
subspace.) -/
def direction (s : AffineSubspace k P) : Submodule k V :=
  vectorSpan k (s : Set P)


/-- The direction equals the `vectorSpan`. -/
theorem direction_eq_vectorSpan (s : AffineSubspace k P) : s.direction = vectorSpan k (s : Set P) :=
  rfl


/-- Alternative definition of the direction when the affine subspace is nonempty. This is defined so
that the order on submodules (as used in the definition of `Submodule.span`) can be used in the
proof of `coe_direction_eq_vsub_set`, and is not intended to be used beyond that proof. -/
def directionOfNonempty {s : AffineSubspace k P} (h : (s : Set P).Nonempty) : Submodule k V where
  carrier := (s : Set P) -ᵥ s
  zero_mem' := by
    /-
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      h : (↑s).Nonempty
      ⊢ Membership.mem { carrier := VSub.vsub ↑s ↑s, add_mem' := ⋯ }.carrier 0
    -/
    cases' h with p hp
    /-
      case intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hp : Membership.mem (↑s) p
      ⊢ Membership.mem { carrier := VSub.vsub ↑s ↑s, add_mem' := ⋯ }.carrier 0
    -/
    /-
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      h : (↑s).Nonempty
      ⊢ ∀ {a b : V}, Membership.mem (VSub.vsub ↑s ↑s) a → Membership.mem (VSub.vsub  …
    -/
    exact vsub_self p ▸ vsub_mem_vsub hp hp
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      h : (↑s).Nonempty
      p1 : P
      hp1 : Membership.mem (↑s) p1
      p2 : P
      hp2 : Membership.mem (↑s) p2
      p3 : P
      hp3 : Membership.mem (↑s) p3
      p4 : P
      hp4 : Membership.mem (↑s) p4
      ⊢ Membership.mem (VSub.vsub ↑s ↑s) (HAdd.hAdd ((fun x1 x2 => VSub.vsub x1 x2)  …
    -/
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      h : (↑s).Nonempty
      p1 : P
      hp1 : Membership.mem (↑s) p1
      p2 : P
      hp2 : Membership.mem (↑s) p2
      p3 : P
      hp3 : Membership.mem (↑s) p3
      p4 : P
      hp4 : Membership.mem (↑s) p4
      ⊢ Membership.mem (VSub.vsub ↑s ↑s) (VSub.vsub (HVAdd.hVAdd ((fun x1 x2 => VSub …
    -/
  add_mem' := by
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      h : (↑s).Nonempty
      p1 : P
      hp1 : Membership.mem (↑s) p1
      p2 : P
      hp2 : Membership.mem (↑s) p2
      p3 : P
      hp3 : Membership.mem (↑s) p3
      p4 : P
      hp4 : Membership.mem (↑s) p4
      ⊢ Membership.mem (↑s) (HVAdd.hVAdd ((fun x1 x2 => VSub.vsub x1 x2) p1 p2) p3)
    -/
    rintro _ _ ⟨p1, hp1, p2, hp2, rfl⟩ ⟨p3, hp3, p4, hp4, rfl⟩
    /-
      case h.e'_5.h.e'_5
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      h : (↑s).Nonempty
      p1 : P
      hp1 : Membership.mem (↑s) p1
      p2 : P
      hp2 : Membership.mem (↑s) p2
      p3 : P
      hp3 : Membership.mem (↑s) p3
      p4 : P
      hp4 : Membership.mem (↑s) p4
      ⊢ Eq ((fun x1 x2 => VSub.vsub x1 x2) p1 p2) (HSMul.hSMul 1 (VSub.vsub p1 p2))
    -/
    rw [← vadd_vsub_assoc]
    /-
      🎉 no goals
    -/
    refine vsub_mem_vsub ?_ hp4
    convert s.smul_vsub_vadd_mem 1 hp1 hp2 hp3
    rw [one_smul]
  smul_mem' := by
    /-
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      h : (↑s).Nonempty
      ⊢ ∀ (c : k) {x : V}, Membership.mem { carrier := VSub.vsub ↑s ↑s, add_mem' :=  …
    -/
    rintro c _ ⟨p1, hp1, p2, hp2, rfl⟩
    /-
      case intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      h : (↑s).Nonempty
      c : k
      p1 : P
      hp1 : Membership.mem (↑s) p1
      p2 : P
      hp2 : Membership.mem (↑s) p2
      ⊢ Membership.mem { carrier := VSub.vsub ↑s ↑s, add_mem' := ⋯, zero_mem' := ⋯ } …
    -/
    rw [← vadd_vsub (c • (p1 -ᵥ p2)) p2]
    /-
      case intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      h : (↑s).Nonempty
      c : k
      p1 : P
      hp1 : Membership.mem (↑s) p1
      p2 : P
      hp2 : Membership.mem (↑s) p2
      ⊢ Membership.mem { carrier := VSub.vsub ↑s ↑s, add_mem' := ⋯, zero_mem' := ⋯ } …
    -/
    refine vsub_mem_vsub ?_ hp2
    /-
      case intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      h : (↑s).Nonempty
      c : k
      p1 : P
      hp1 : Membership.mem (↑s) p1
      p2 : P
      hp2 : Membership.mem (↑s) p2
      ⊢ Membership.mem (↑s) (HVAdd.hVAdd (HSMul.hSMul c (VSub.vsub p1 p2)) p2)
    -/
    exact s.smul_vsub_vadd_mem c hp1 hp2 hp2
    /-
      🎉 no goals
    -/


/-- `direction_of_nonempty` gives the same submodule as `direction`. -/
theorem directionOfNonempty_eq_direction {s : AffineSubspace k P} (h : (s : Set P).Nonempty) :
    directionOfNonempty h = s.direction := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    h : (↑s).Nonempty
    ⊢ Eq (AffineSubspace.directionOfNonempty h) s.direction
  -/
  refine le_antisymm ?_ (Submodule.span_le.2 Set.Subset.rfl)
  rw [← SetLike.coe_subset_coe, directionOfNonempty, direction, Submodule.coe_set_mk,
    AddSubmonoid.coe_set_mk]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    h : (↑s).Nonempty
    ⊢ HasSubset.Subset ↑{ carrier := VSub.vsub ↑s ↑s, add_mem' := ⋯ } ↑(vectorSpan …
  -/
  exact vsub_set_subset_vectorSpan k _
  /-
    🎉 no goals
  -/


/-- The set of vectors in the direction of a nonempty affine subspace is given by `vsub_set`. -/
theorem coe_direction_eq_vsub_set {s : AffineSubspace k P} (h : (s : Set P).Nonempty) :
    (s.direction : Set V) = (s : Set P) -ᵥ s :=
  directionOfNonempty_eq_direction h ▸ rfl


/-- A vector is in the direction of a nonempty affine subspace if and only if it is the subtraction
of two vectors in the subspace. -/
theorem mem_direction_iff_eq_vsub {s : AffineSubspace k P} (h : (s : Set P).Nonempty) (v : V) :
    v ∈ s.direction ↔ ∃ p1 ∈ s, ∃ p2 ∈ s, v = p1 -ᵥ p2 := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    h : (↑s).Nonempty
    v : V
    ⊢ Iff (Membership.mem s.direction v) (Exists fun p1 => And (Membership.mem s p …
  -/
  rw [← SetLike.mem_coe, coe_direction_eq_vsub_set h, Set.mem_vsub]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    h : (↑s).Nonempty
    v : V
    ⊢ Iff (Exists fun x => And (Membership.mem (↑s) x) (Exists fun y => And (Membe …
  -/
  simp only [SetLike.mem_coe, eq_comm]
  /-
    🎉 no goals
  -/


/-- Adding a vector in the direction to a point in the subspace produces a point in the
subspace. -/
theorem vadd_mem_of_mem_direction {s : AffineSubspace k P} {v : V} (hv : v ∈ s.direction) {p : P}
    (hp : p ∈ s) : v +ᵥ p ∈ s := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    v : V
    hv : Membership.mem s.direction v
    p : P
    hp : Membership.mem s p
    ⊢ Membership.mem s (HVAdd.hVAdd v p)
  -/
  rw [mem_direction_iff_eq_vsub ⟨p, hp⟩] at hv
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    v : V
    hv : Exists fun p1 => And (Membership.mem s p1) (Exists fun p2 => And (Members …
    p : P
    hp : Membership.mem s p
    ⊢ Membership.mem s (HVAdd.hVAdd v p)
  -/
  rcases hv with ⟨p1, hp1, p2, hp2, hv⟩
  /-
    case intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    v : V
    p : P
    hp : Membership.mem s p
    p1 : P
    hp1 : Membership.mem s p1
    p2 : P
    hp2 : Membership.mem s p2
    hv : Eq v (VSub.vsub p1 p2)
    ⊢ Membership.mem s (HVAdd.hVAdd v p)
  -/
  rw [hv]
  /-
    case intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    v : V
    p : P
    hp : Membership.mem s p
    p1 : P
    hp1 : Membership.mem s p1
    p2 : P
    hp2 : Membership.mem s p2
    hv : Eq v (VSub.vsub p1 p2)
    ⊢ Membership.mem s (HVAdd.hVAdd (VSub.vsub p1 p2) p)
  -/
  convert s.smul_vsub_vadd_mem 1 hp1 hp2 hp
  /-
    case h.e'_1.h.e'_5
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    v : V
    p : P
    hp : Membership.mem s p
    p1 : P
    hp1 : Membership.mem s p1
    p2 : P
    hp2 : Membership.mem s p2
    hv : Eq v (VSub.vsub p1 p2)
    ⊢ Eq (VSub.vsub p1 p2) (HSMul.hSMul 1 (VSub.vsub p1 p2))
  -/
  rw [one_smul]
  /-
    🎉 no goals
  -/


/-- Subtracting two points in the subspace produces a vector in the direction. -/
theorem vsub_mem_direction {s : AffineSubspace k P} {p1 p2 : P} (hp1 : p1 ∈ s) (hp2 : p2 ∈ s) :
    p1 -ᵥ p2 ∈ s.direction :=
  vsub_mem_vectorSpan k hp1 hp2


/-- Adding a vector to a point in a subspace produces a point in the subspace if and only if the
vector is in the direction. -/
theorem vadd_mem_iff_mem_direction {s : AffineSubspace k P} (v : V) {p : P} (hp : p ∈ s) :
    v +ᵥ p ∈ s ↔ v ∈ s.direction :=
               /-
                 k : Type u_1
                 V : Type u_2
                 P : Type u_3
                 inst✝³ : Ring k
                 inst✝² : AddCommGroup V
                 inst✝¹ : Module k V
                 inst✝ : AddTorsor V P
                 s : AffineSubspace k P
                 v : V
                 p : P
                 hp : Membership.mem s p
                 h : Membership.mem s (HVAdd.hVAdd v p)
                 ⊢ Membership.mem s.direction v
               -/
  ⟨fun h => by simpa using vsub_mem_direction h hp, fun h => vadd_mem_of_mem_direction h hp⟩
               /-
                 🎉 no goals
               -/


/-- Adding a vector in the direction to a point produces a point in the subspace if and only if
the original point is in the subspace. -/
theorem vadd_mem_iff_mem_of_mem_direction {s : AffineSubspace k P} {v : V} (hv : v ∈ s.direction)
    {p : P} : v +ᵥ p ∈ s ↔ p ∈ s := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    v : V
    hv : Membership.mem s.direction v
    p : P
    ⊢ Iff (Membership.mem s (HVAdd.hVAdd v p)) (Membership.mem s p)
  -/
  refine ⟨fun h => ?_, fun h => vadd_mem_of_mem_direction hv h⟩
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    v : V
    hv : Membership.mem s.direction v
    p : P
    h : Membership.mem s (HVAdd.hVAdd v p)
    ⊢ Membership.mem s p
  -/
  convert vadd_mem_of_mem_direction (Submodule.neg_mem _ hv) h
  /-
    case h.e'_5
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    v : V
    hv : Membership.mem s.direction v
    p : P
    h : Membership.mem s (HVAdd.hVAdd v p)
    ⊢ Eq p (HVAdd.hVAdd (Neg.neg v) (HVAdd.hVAdd v p))
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given a point in an affine subspace, the set of vectors in its direction equals the set of
vectors subtracting that point on the right. -/
theorem coe_direction_eq_vsub_set_right {s : AffineSubspace k P} {p : P} (hp : p ∈ s) :
    (s.direction : Set V) = (· -ᵥ p) '' s := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p : P
    hp : Membership.mem s p
    ⊢ Eq (↑s.direction) (Set.image (fun x => VSub.vsub x p) ↑s)
  -/
  rw [coe_direction_eq_vsub_set ⟨p, hp⟩]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p : P
    hp : Membership.mem s p
    ⊢ Eq (VSub.vsub ↑s ↑s) (Set.image (fun x => VSub.vsub x p) ↑s)
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hp : Membership.mem s p
      ⊢ LE.le (VSub.vsub ↑s ↑s) (Set.image (fun x => VSub.vsub x p) ↑s)
    -/
  · rintro v ⟨p1, hp1, p2, hp2, rfl⟩
    exact ⟨(p1 -ᵥ p2) +ᵥ p,
      vadd_mem_of_mem_direction (vsub_mem_direction hp1 hp2) hp, vadd_vsub _ _⟩
    /-
      case refine_2
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hp : Membership.mem s p
      ⊢ LE.le (Set.image (fun x => VSub.vsub x p) ↑s) (VSub.vsub ↑s ↑s)
    -/
  · rintro v ⟨p2, hp2, rfl⟩
    /-
      case refine_2.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p : P
      hp : Membership.mem s p
      p2 : P
      hp2 : Membership.mem (↑s) p2
      ⊢ Membership.mem (VSub.vsub ↑s ↑s) ((fun x => VSub.vsub x p) p2)
    -/
    exact ⟨p2, hp2, p, hp, rfl⟩
    /-
      🎉 no goals
    -/


/-- Given a point in an affine subspace, the set of vectors in its direction equals the set of
vectors subtracting that point on the left. -/
theorem coe_direction_eq_vsub_set_left {s : AffineSubspace k P} {p : P} (hp : p ∈ s) :
    (s.direction : Set V) = (p -ᵥ ·) '' s := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p : P
    hp : Membership.mem s p
    ⊢ Eq (↑s.direction) (Set.image (fun x => VSub.vsub p x) ↑s)
  -/
  ext v
  rw [SetLike.mem_coe, ← Submodule.neg_mem_iff, ← SetLike.mem_coe,
    coe_direction_eq_vsub_set_right hp, Set.mem_image, Set.mem_image]
  conv_lhs =>
    congr
    ext
    rw [← neg_vsub_eq_vsub_rev, neg_inj]


/-- Given a point in an affine subspace, a vector is in its direction if and only if it results from
subtracting that point on the right. -/
theorem mem_direction_iff_eq_vsub_right {s : AffineSubspace k P} {p : P} (hp : p ∈ s) (v : V) :
    v ∈ s.direction ↔ ∃ p2 ∈ s, v = p2 -ᵥ p := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p : P
    hp : Membership.mem s p
    v : V
    ⊢ Iff (Membership.mem s.direction v) (Exists fun p2 => And (Membership.mem s p …
  -/
  rw [← SetLike.mem_coe, coe_direction_eq_vsub_set_right hp]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p : P
    hp : Membership.mem s p
    v : V
    ⊢ Iff (Membership.mem (Set.image (fun x => VSub.vsub x p) ↑s) v) (Exists fun p …
  -/
  exact ⟨fun ⟨p2, hp2, hv⟩ => ⟨p2, hp2, hv.symm⟩, fun ⟨p2, hp2, hv⟩ => ⟨p2, hp2, hv.symm⟩⟩
  /-
    🎉 no goals
  -/


/-- Given a point in an affine subspace, a vector is in its direction if and only if it results from
subtracting that point on the left. -/
theorem mem_direction_iff_eq_vsub_left {s : AffineSubspace k P} {p : P} (hp : p ∈ s) (v : V) :
    v ∈ s.direction ↔ ∃ p2 ∈ s, v = p -ᵥ p2 := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p : P
    hp : Membership.mem s p
    v : V
    ⊢ Iff (Membership.mem s.direction v) (Exists fun p2 => And (Membership.mem s p …
  -/
  rw [← SetLike.mem_coe, coe_direction_eq_vsub_set_left hp]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p : P
    hp : Membership.mem s p
    v : V
    ⊢ Iff (Membership.mem (Set.image (fun x => VSub.vsub p x) ↑s) v) (Exists fun p …
  -/
  exact ⟨fun ⟨p2, hp2, hv⟩ => ⟨p2, hp2, hv.symm⟩, fun ⟨p2, hp2, hv⟩ => ⟨p2, hp2, hv.symm⟩⟩
  /-
    🎉 no goals
  -/


/-- Given a point in an affine subspace, a result of subtracting that point on the right is in the
direction if and only if the other point is in the subspace. -/
theorem vsub_right_mem_direction_iff_mem {s : AffineSubspace k P} {p : P} (hp : p ∈ s) (p2 : P) :
    p2 -ᵥ p ∈ s.direction ↔ p2 ∈ s := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p : P
    hp : Membership.mem s p
    p2 : P
    ⊢ Iff (Membership.mem s.direction (VSub.vsub p2 p)) (Membership.mem s p2)
  -/
  rw [mem_direction_iff_eq_vsub_right hp]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p : P
    hp : Membership.mem s p
    p2 : P
    ⊢ Iff (Exists fun p2_1 => And (Membership.mem s p2_1) (Eq (VSub.vsub p2 p) (VS …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given a point in an affine subspace, a result of subtracting that point on the left is in the
direction if and only if the other point is in the subspace. -/
theorem vsub_left_mem_direction_iff_mem {s : AffineSubspace k P} {p : P} (hp : p ∈ s) (p2 : P) :
    p -ᵥ p2 ∈ s.direction ↔ p2 ∈ s := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p : P
    hp : Membership.mem s p
    p2 : P
    ⊢ Iff (Membership.mem s.direction (VSub.vsub p p2)) (Membership.mem s p2)
  -/
  rw [mem_direction_iff_eq_vsub_left hp]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p : P
    hp : Membership.mem s p
    p2 : P
    ⊢ Iff (Exists fun p2_1 => And (Membership.mem s p2_1) (Eq (VSub.vsub p p2) (VS …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Two affine subspaces are equal if they have the same points. -/
theorem coe_injective : Function.Injective ((↑) : AffineSubspace k P → Set P) :=
  SetLike.coe_injective


@[ext (iff := false)]
theorem ext {p q : AffineSubspace k P} (h : ∀ x, x ∈ p ↔ x ∈ q) : p = q :=
  SetLike.ext h


protected theorem ext_iff (s₁ s₂ : AffineSubspace k P) : s₁ = s₂ ↔ (s₁ : Set P) = s₂ :=
  SetLike.ext'_iff


/-- Two affine subspaces with the same direction and nonempty intersection are equal. -/
theorem ext_of_direction_eq {s1 s2 : AffineSubspace k P} (hd : s1.direction = s2.direction)
    (hn : ((s1 : Set P) ∩ s2).Nonempty) : s1 = s2 := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s1 s2 : AffineSubspace k P
    hd : Eq s1.direction s2.direction
    hn : (Inter.inter ↑s1 ↑s2).Nonempty
    ⊢ Eq s1 s2
  -/
  ext p
  /-
    case h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s1 s2 : AffineSubspace k P
    hd : Eq s1.direction s2.direction
    hn : (Inter.inter ↑s1 ↑s2).Nonempty
    p : P
    ⊢ Iff (Membership.mem s1 p) (Membership.mem s2 p)
  -/
  have hq1 := Set.mem_of_mem_inter_left hn.some_mem
  /-
    case h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s1 s2 : AffineSubspace k P
    hd : Eq s1.direction s2.direction
    hn : (Inter.inter ↑s1 ↑s2).Nonempty
    p : P
    hq1 : Membership.mem (↑s1) hn.some
    ⊢ Iff (Membership.mem s1 p) (Membership.mem s2 p)
  -/
  have hq2 := Set.mem_of_mem_inter_right hn.some_mem
  /-
    case h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s1 s2 : AffineSubspace k P
    hd : Eq s1.direction s2.direction
    hn : (Inter.inter ↑s1 ↑s2).Nonempty
    p : P
    hq1 : Membership.mem (↑s1) hn.some
    hq2 : Membership.mem (↑s2) hn.some
    ⊢ Iff (Membership.mem s1 p) (Membership.mem s2 p)
  -/
  constructor
    /-
      case h.mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s1 s2 : AffineSubspace k P
      hd : Eq s1.direction s2.direction
      hn : (Inter.inter ↑s1 ↑s2).Nonempty
      p : P
      hq1 : Membership.mem (↑s1) hn.some
      hq2 : Membership.mem (↑s2) hn.some
      ⊢ Membership.mem s1 p → Membership.mem s2 p
    -/
  · intro hp
    /-
      case h.mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s1 s2 : AffineSubspace k P
      hd : Eq s1.direction s2.direction
      hn : (Inter.inter ↑s1 ↑s2).Nonempty
      p : P
      hq1 : Membership.mem (↑s1) hn.some
      hq2 : Membership.mem (↑s2) hn.some
      hp : Membership.mem s1 p
      ⊢ Membership.mem s2 p
    -/
    rw [← vsub_vadd p hn.some]
    /-
      case h.mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s1 s2 : AffineSubspace k P
      hd : Eq s1.direction s2.direction
      hn : (Inter.inter ↑s1 ↑s2).Nonempty
      p : P
      hq1 : Membership.mem (↑s1) hn.some
      hq2 : Membership.mem (↑s2) hn.some
      hp : Membership.mem s1 p
      ⊢ Membership.mem s2 (HVAdd.hVAdd (VSub.vsub p hn.some) hn.some)
    -/
    refine vadd_mem_of_mem_direction ?_ hq2
    /-
      case h.mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s1 s2 : AffineSubspace k P
      hd : Eq s1.direction s2.direction
      hn : (Inter.inter ↑s1 ↑s2).Nonempty
      p : P
      hq1 : Membership.mem (↑s1) hn.some
      hq2 : Membership.mem (↑s2) hn.some
      hp : Membership.mem s1 p
      ⊢ Membership.mem s2.direction (VSub.vsub p hn.some)
    -/
    rw [← hd]
    /-
      case h.mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s1 s2 : AffineSubspace k P
      hd : Eq s1.direction s2.direction
      hn : (Inter.inter ↑s1 ↑s2).Nonempty
      p : P
      hq1 : Membership.mem (↑s1) hn.some
      hq2 : Membership.mem (↑s2) hn.some
      hp : Membership.mem s1 p
      ⊢ Membership.mem s1.direction (VSub.vsub p hn.some)
    -/
    exact vsub_mem_direction hp hq1
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s1 s2 : AffineSubspace k P
      hd : Eq s1.direction s2.direction
      hn : (Inter.inter ↑s1 ↑s2).Nonempty
      p : P
      hq1 : Membership.mem (↑s1) hn.some
      hq2 : Membership.mem (↑s2) hn.some
      ⊢ Membership.mem s2 p → Membership.mem s1 p
    -/
  · intro hp
    /-
      case h.mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s1 s2 : AffineSubspace k P
      hd : Eq s1.direction s2.direction
      hn : (Inter.inter ↑s1 ↑s2).Nonempty
      p : P
      hq1 : Membership.mem (↑s1) hn.some
      hq2 : Membership.mem (↑s2) hn.some
      hp : Membership.mem s2 p
      ⊢ Membership.mem s1 p
    -/
    rw [← vsub_vadd p hn.some]
    /-
      case h.mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s1 s2 : AffineSubspace k P
      hd : Eq s1.direction s2.direction
      hn : (Inter.inter ↑s1 ↑s2).Nonempty
      p : P
      hq1 : Membership.mem (↑s1) hn.some
      hq2 : Membership.mem (↑s2) hn.some
      hp : Membership.mem s2 p
      ⊢ Membership.mem s1 (HVAdd.hVAdd (VSub.vsub p hn.some) hn.some)
    -/
    refine vadd_mem_of_mem_direction ?_ hq1
    /-
      case h.mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s1 s2 : AffineSubspace k P
      hd : Eq s1.direction s2.direction
      hn : (Inter.inter ↑s1 ↑s2).Nonempty
      p : P
      hq1 : Membership.mem (↑s1) hn.some
      hq2 : Membership.mem (↑s2) hn.some
      hp : Membership.mem s2 p
      ⊢ Membership.mem s1.direction (VSub.vsub p hn.some)
    -/
    rw [hd]
    /-
      case h.mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s1 s2 : AffineSubspace k P
      hd : Eq s1.direction s2.direction
      hn : (Inter.inter ↑s1 ↑s2).Nonempty
      p : P
      hq1 : Membership.mem (↑s1) hn.some
      hq2 : Membership.mem (↑s2) hn.some
      hp : Membership.mem s2 p
      ⊢ Membership.mem s2.direction (VSub.vsub p hn.some)
    -/
    exact vsub_mem_direction hp hq2
    /-
      🎉 no goals
    -/

-- See note [reducible non instances]

/-- This is not an instance because it loops with `AddTorsor.nonempty`. -/
abbrev toAddTorsor (s : AffineSubspace k P) [Nonempty s] : AddTorsor s.direction s where
  vadd a b := ⟨(a : V) +ᵥ (b : P), vadd_mem_of_mem_direction a.2 b.2⟩
  zero_vadd := fun a => by
    /-
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      s : AffineSubspace k P
      inst✝ : Nonempty (Subtype fun x => Membership.mem s x)
      a : Subtype fun x => Membership.mem s x
      ⊢ Eq (HVAdd.hVAdd 0 a) a
    -/
    ext
    /-
      case a
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      s : AffineSubspace k P
      inst✝ : Nonempty (Subtype fun x => Membership.mem s x)
      a : Subtype fun x => Membership.mem s x
      ⊢ Eq ↑(HVAdd.hVAdd 0 a) ↑a
    -/
    exact zero_vadd _ _
    /-
      🎉 no goals
    -/
  add_vadd a b c := by
    /-
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      s : AffineSubspace k P
      inst✝ : Nonempty (Subtype fun x => Membership.mem s x)
      a b : Subtype fun x => Membership.mem s.direction x
      c : Subtype fun x => Membership.mem s x
      ⊢ Eq (HVAdd.hVAdd (HAdd.hAdd a b) c) (HVAdd.hVAdd a (HVAdd.hVAdd b c))
    -/
    ext
    /-
      case a
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      s : AffineSubspace k P
      inst✝ : Nonempty (Subtype fun x => Membership.mem s x)
      a b : Subtype fun x => Membership.mem s.direction x
      c : Subtype fun x => Membership.mem s x
      ⊢ Eq ↑(HVAdd.hVAdd (HAdd.hAdd a b) c) ↑(HVAdd.hVAdd a (HVAdd.hVAdd b c))
    -/
    apply add_vadd
    /-
      🎉 no goals
    -/
  vsub a b := ⟨(a : P) -ᵥ (b : P), (vsub_left_mem_direction_iff_mem a.2 _).mpr b.2⟩
  vsub_vadd' a b := by
    /-
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      s : AffineSubspace k P
      inst✝ : Nonempty (Subtype fun x => Membership.mem s x)
      a b : Subtype fun x => Membership.mem s x
      ⊢ Eq (HVAdd.hVAdd (VSub.vsub a b) b) a
    -/
    ext
    /-
      case a
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      s : AffineSubspace k P
      inst✝ : Nonempty (Subtype fun x => Membership.mem s x)
      a b : Subtype fun x => Membership.mem s x
      ⊢ Eq ↑(HVAdd.hVAdd (VSub.vsub a b) b) ↑a
    -/
    apply AddTorsor.vsub_vadd'
    /-
      🎉 no goals
    -/
  vadd_vsub' a b := by
    /-
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      s : AffineSubspace k P
      inst✝ : Nonempty (Subtype fun x => Membership.mem s x)
      a : Subtype fun x => Membership.mem s.direction x
      b : Subtype fun x => Membership.mem s x
      ⊢ Eq (VSub.vsub (HVAdd.hVAdd a b) b) a
    -/
    ext
    /-
      case a
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      s : AffineSubspace k P
      inst✝ : Nonempty (Subtype fun x => Membership.mem s x)
      a : Subtype fun x => Membership.mem s.direction x
      b : Subtype fun x => Membership.mem s x
      ⊢ Eq ↑(VSub.vsub (HVAdd.hVAdd a b) b) ↑a
    -/
    apply AddTorsor.vadd_vsub'
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem coe_vsub (s : AffineSubspace k P) [Nonempty s] (a b : s) : ↑(a -ᵥ b) = (a : P) -ᵥ (b : P) :=
  rfl


@[simp, norm_cast]
theorem coe_vadd (s : AffineSubspace k P) [Nonempty s] (a : s.direction) (b : s) :
    ↑(a +ᵥ b) = (a : V) +ᵥ (b : P) :=
  rfl


/-- Embedding of an affine subspace to the ambient space, as an affine map. -/
protected def subtype (s : AffineSubspace k P) [Nonempty s] : s →ᵃ[k] P where
  toFun := (↑)
  linear := s.direction.subtype
  map_vadd' _ _ := rfl


@[simp]
theorem subtype_linear (s : AffineSubspace k P) [Nonempty s] :
    s.subtype.linear = s.direction.subtype := rfl


theorem subtype_apply (s : AffineSubspace k P) [Nonempty s] (p : s) : s.subtype p = p :=
  rfl


@[simp]
theorem coeSubtype (s : AffineSubspace k P) [Nonempty s] : (s.subtype : s → P) = ((↑) : s → P) :=
  rfl


theorem injective_subtype (s : AffineSubspace k P) [Nonempty s] : Function.Injective s.subtype :=
  Subtype.coe_injective


/-- Two affine subspaces with nonempty intersection are equal if and only if their directions are
equal. -/
theorem eq_iff_direction_eq_of_mem {s₁ s₂ : AffineSubspace k P} {p : P} (h₁ : p ∈ s₁)
    (h₂ : p ∈ s₂) : s₁ = s₂ ↔ s₁.direction = s₂.direction :=
  ⟨fun h => h ▸ rfl, fun h => ext_of_direction_eq h ⟨p, h₁, h₂⟩⟩


/-- Construct an affine subspace from a point and a direction. -/
def mk' (p : P) (direction : Submodule k V) : AffineSubspace k P where
  carrier := { q | ∃ v ∈ direction, q = v +ᵥ p }
  smul_vsub_vadd_mem c p1 p2 p3 hp1 hp2 hp3 := by
    /-
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p : P
      direction : Submodule k V
      c : k
      p1 p2 p3 : P
      hp1 : Membership.mem (setOf fun q => Exists fun v => And (Membership.mem direc …
      hp2 : Membership.mem (setOf fun q => Exists fun v => And (Membership.mem direc …
      hp3 : Membership.mem (setOf fun q => Exists fun v => And (Membership.mem direc …
      ⊢ Membership.mem (setOf fun q => Exists fun v => And (Membership.mem direction …
    -/
    rcases hp1 with ⟨v1, hv1, hp1⟩
    /-
      case intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p : P
      direction : Submodule k V
      c : k
      p1 p2 p3 : P
      hp2 : Membership.mem (setOf fun q => Exists fun v => And (Membership.mem direc …
      hp3 : Membership.mem (setOf fun q => Exists fun v => And (Membership.mem direc …
      v1 : V
      hv1 : Membership.mem direction v1
      hp1 : Eq p1 (HVAdd.hVAdd v1 p)
      ⊢ Membership.mem (setOf fun q => Exists fun v => And (Membership.mem direction …
    -/
    rcases hp2 with ⟨v2, hv2, hp2⟩
    /-
      case intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p : P
      direction : Submodule k V
      c : k
      p1 p2 p3 : P
      hp3 : Membership.mem (setOf fun q => Exists fun v => And (Membership.mem direc …
      v1 : V
      hv1 : Membership.mem direction v1
      hp1 : Eq p1 (HVAdd.hVAdd v1 p)
      v2 : V
      hv2 : Membership.mem direction v2
      hp2 : Eq p2 (HVAdd.hVAdd v2 p)
      ⊢ Membership.mem (setOf fun q => Exists fun v => And (Membership.mem direction …
    -/
    rcases hp3 with ⟨v3, hv3, hp3⟩
    /-
      case intro.intro.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p : P
      direction : Submodule k V
      c : k
      p1 p2 p3 : P
      v1 : V
      hv1 : Membership.mem direction v1
      hp1 : Eq p1 (HVAdd.hVAdd v1 p)
      v2 : V
      hv2 : Membership.mem direction v2
      hp2 : Eq p2 (HVAdd.hVAdd v2 p)
      v3 : V
      hv3 : Membership.mem direction v3
      hp3 : Eq p3 (HVAdd.hVAdd v3 p)
      ⊢ Membership.mem (setOf fun q => Exists fun v => And (Membership.mem direction …
    -/
    use c • (v1 - v2) + v3, direction.add_mem (direction.smul_mem c (direction.sub_mem hv1 hv2)) hv3
    /-
      case right
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p : P
      direction : Submodule k V
      c : k
      p1 p2 p3 : P
      v1 : V
      hv1 : Membership.mem direction v1
      hp1 : Eq p1 (HVAdd.hVAdd v1 p)
      v2 : V
      hv2 : Membership.mem direction v2
      hp2 : Eq p2 (HVAdd.hVAdd v2 p)
      v3 : V
      hv3 : Membership.mem direction v3
      hp3 : Eq p3 (HVAdd.hVAdd v3 p)
      ⊢ Eq (HVAdd.hVAdd (HSMul.hSMul c (VSub.vsub p1 p2)) p3) (HVAdd.hVAdd (HAdd.hAd …
    -/
    simp [hp1, hp2, hp3, vadd_vadd]
    /-
      🎉 no goals
    -/


/-- An affine subspace constructed from a point and a direction contains that point. -/
theorem self_mem_mk' (p : P) (direction : Submodule k V) : p ∈ mk' p direction :=
  ⟨0, ⟨direction.zero_mem, (zero_vadd _ _).symm⟩⟩


/-- An affine subspace constructed from a point and a direction contains the result of adding a
vector in that direction to that point. -/
theorem vadd_mem_mk' {v : V} (p : P) {direction : Submodule k V} (hv : v ∈ direction) :
    v +ᵥ p ∈ mk' p direction :=
  ⟨v, hv, rfl⟩


/-- An affine subspace constructed from a point and a direction is nonempty. -/
theorem mk'_nonempty (p : P) (direction : Submodule k V) : (mk' p direction : Set P).Nonempty :=
  ⟨p, self_mem_mk' p direction⟩


/-- The direction of an affine subspace constructed from a point and a direction. -/
@[simp]
theorem direction_mk' (p : P) (direction : Submodule k V) :
    (mk' p direction).direction = direction := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p : P
    direction : Submodule k V
    ⊢ Eq (AffineSubspace.mk' p direction).direction direction
  -/
  ext v
  /-
    case h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p : P
    direction : Submodule k V
    v : V
    ⊢ Iff (Membership.mem (AffineSubspace.mk' p direction).direction v) (Membershi …
  -/
  rw [mem_direction_iff_eq_vsub (mk'_nonempty _ _)]
  /-
    case h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p : P
    direction : Submodule k V
    v : V
    ⊢ Iff (Exists fun p1 => And (Membership.mem (AffineSubspace.mk' p direction) p …
  -/
  constructor
    /-
      case h.mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p : P
      direction : Submodule k V
      v : V
      ⊢ (Exists fun p1 => And (Membership.mem (AffineSubspace.mk' p direction) p1) ( …
    -/
  · rintro ⟨p1, ⟨v1, hv1, hp1⟩, p2, ⟨v2, hv2, hp2⟩, hv⟩
    /-
      case h.mp.intro.intro.intro.intro.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p : P
      direction : Submodule k V
      v : V
      p1 : P
      v1 : V
      hv1 : Membership.mem direction v1
      hp1 : Eq p1 (HVAdd.hVAdd v1 p)
      p2 : P
      hv : Eq v (VSub.vsub p1 p2)
      v2 : V
      hv2 : Membership.mem direction v2
      hp2 : Eq p2 (HVAdd.hVAdd v2 p)
      ⊢ Membership.mem direction v
    -/
    rw [hv, hp1, hp2, vadd_vsub_vadd_cancel_right]
    /-
      case h.mp.intro.intro.intro.intro.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p : P
      direction : Submodule k V
      v : V
      p1 : P
      v1 : V
      hv1 : Membership.mem direction v1
      hp1 : Eq p1 (HVAdd.hVAdd v1 p)
      p2 : P
      hv : Eq v (VSub.vsub p1 p2)
      v2 : V
      hv2 : Membership.mem direction v2
      hp2 : Eq p2 (HVAdd.hVAdd v2 p)
      ⊢ Membership.mem direction (HSub.hSub v1 v2)
    -/
    exact direction.sub_mem hv1 hv2
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p : P
      direction : Submodule k V
      v : V
      ⊢ Membership.mem direction v → Exists fun p1 => And (Membership.mem (AffineSub …
    -/
  · exact fun hv => ⟨v +ᵥ p, vadd_mem_mk' _ hv, p, self_mem_mk' _ _, (vadd_vsub _ _).symm⟩
    /-
      🎉 no goals
    -/


/-- A point lies in an affine subspace constructed from another point and a direction if and only
if their difference is in that direction. -/
theorem mem_mk'_iff_vsub_mem {p₁ p₂ : P} {direction : Submodule k V} :
    p₂ ∈ mk' p₁ direction ↔ p₂ -ᵥ p₁ ∈ direction := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ : P
    direction : Submodule k V
    ⊢ Iff (Membership.mem (AffineSubspace.mk' p₁ direction) p₂) (Membership.mem di …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p₁ p₂ : P
      direction : Submodule k V
      h : Membership.mem (AffineSubspace.mk' p₁ direction) p₂
      ⊢ Membership.mem direction (VSub.vsub p₂ p₁)
    -/
  · rw [← direction_mk' p₁ direction]
    /-
      case refine_1
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p₁ p₂ : P
      direction : Submodule k V
      h : Membership.mem (AffineSubspace.mk' p₁ direction) p₂
      ⊢ Membership.mem (AffineSubspace.mk' p₁ direction).direction (VSub.vsub p₂ p₁)
    -/
    exact vsub_mem_direction h (self_mem_mk' _ _)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p₁ p₂ : P
      direction : Submodule k V
      h : Membership.mem direction (VSub.vsub p₂ p₁)
      ⊢ Membership.mem (AffineSubspace.mk' p₁ direction) p₂
    -/
  · rw [← vsub_vadd p₂ p₁]
    /-
      case refine_2
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      p₁ p₂ : P
      direction : Submodule k V
      h : Membership.mem direction (VSub.vsub p₂ p₁)
      ⊢ Membership.mem (AffineSubspace.mk' p₁ direction) (HVAdd.hVAdd (VSub.vsub p₂  …
    -/
    exact vadd_mem_mk' p₁ h
    /-
      🎉 no goals
    -/


/-- Constructing an affine subspace from a point in a subspace and that subspace's direction
yields the original subspace. -/
@[simp]
theorem mk'_eq {s : AffineSubspace k P} {p : P} (hp : p ∈ s) : mk' p s.direction = s :=
  ext_of_direction_eq (direction_mk' p s.direction) ⟨p, Set.mem_inter (self_mem_mk' _ _) hp⟩


/-- If an affine subspace contains a set of points, it contains the `spanPoints` of that set. -/
theorem spanPoints_subset_coe_of_subset_coe {s : Set P} {s1 : AffineSubspace k P} (h : s ⊆ s1) :
    spanPoints k s ⊆ s1 := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    s1 : AffineSubspace k P
    h : HasSubset.Subset s ↑s1
    ⊢ HasSubset.Subset (spanPoints k s) ↑s1
  -/
  rintro p ⟨p1, hp1, v, hv, hp⟩
  /-
    case intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    s1 : AffineSubspace k P
    h : HasSubset.Subset s ↑s1
    p p1 : P
    hp1 : Membership.mem s p1
    v : V
    hv : Membership.mem (vectorSpan k s) v
    hp : Eq p (HVAdd.hVAdd v p1)
    ⊢ Membership.mem (↑s1) p
  -/
  rw [hp]
  /-
    case intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    s1 : AffineSubspace k P
    h : HasSubset.Subset s ↑s1
    p p1 : P
    hp1 : Membership.mem s p1
    v : V
    hv : Membership.mem (vectorSpan k s) v
    hp : Eq p (HVAdd.hVAdd v p1)
    ⊢ Membership.mem (↑s1) (HVAdd.hVAdd v p1)
  -/
  have hp1s1 : p1 ∈ (s1 : Set P) := Set.mem_of_mem_of_subset hp1 h
  /-
    case intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    s1 : AffineSubspace k P
    h : HasSubset.Subset s ↑s1
    p p1 : P
    hp1 : Membership.mem s p1
    v : V
    hv : Membership.mem (vectorSpan k s) v
    hp : Eq p (HVAdd.hVAdd v p1)
    hp1s1 : Membership.mem (↑s1) p1
    ⊢ Membership.mem (↑s1) (HVAdd.hVAdd v p1)
  -/
  refine vadd_mem_of_mem_direction ?_ hp1s1
  /-
    case intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    s1 : AffineSubspace k P
    h : HasSubset.Subset s ↑s1
    p p1 : P
    hp1 : Membership.mem s p1
    v : V
    hv : Membership.mem (vectorSpan k s) v
    hp : Eq p (HVAdd.hVAdd v p1)
    hp1s1 : Membership.mem (↑s1) p1
    ⊢ Membership.mem s1.direction v
  -/
  have hs : vectorSpan k s ≤ s1.direction := vectorSpan_mono k h
  /-
    case intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    s1 : AffineSubspace k P
    h : HasSubset.Subset s ↑s1
    p p1 : P
    hp1 : Membership.mem s p1
    v : V
    hv : Membership.mem (vectorSpan k s) v
    hp : Eq p (HVAdd.hVAdd v p1)
    hp1s1 : Membership.mem (↑s1) p1
    hs : LE.le (vectorSpan k s) s1.direction
    ⊢ Membership.mem s1.direction v
  -/
  rw [SetLike.le_def] at hs
  /-
    case intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    s1 : AffineSubspace k P
    h : HasSubset.Subset s ↑s1
    p p1 : P
    hp1 : Membership.mem s p1
    v : V
    hv : Membership.mem (vectorSpan k s) v
    hp : Eq p (HVAdd.hVAdd v p1)
    hp1s1 : Membership.mem (↑s1) p1
    hs : ∀ ⦃x : V⦄, Membership.mem (vectorSpan k s) x → Membership.mem s1.directio …
    ⊢ Membership.mem s1.direction v
  -/
  rw [← SetLike.mem_coe]
  /-
    case intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    s1 : AffineSubspace k P
    h : HasSubset.Subset s ↑s1
    p p1 : P
    hp1 : Membership.mem s p1
    v : V
    hv : Membership.mem (vectorSpan k s) v
    hp : Eq p (HVAdd.hVAdd v p1)
    hp1s1 : Membership.mem (↑s1) p1
    hs : ∀ ⦃x : V⦄, Membership.mem (vectorSpan k s) x → Membership.mem s1.directio …
    ⊢ Membership.mem (↑s1.direction) v
  -/
  exact Set.mem_of_mem_of_subset hv hs
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_toAffineSubspace {p : Submodule k V} {x : V} :
    x ∈ p.toAffineSubspace ↔ x ∈ p :=
  Iff.rfl


@[simp]
theorem toAffineSubspace_direction (s : Submodule k V) : s.toAffineSubspace.direction = s := by
  /-
    k : Type u_1
    V : Type u_2
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s : Submodule k V
    ⊢ Eq s.toAffineSubspace.direction s
  -/
  ext x; simp [← s.toAffineSubspace.vadd_mem_iff_mem_direction _ s.zero_mem]
         /-
           🎉 no goals
         -/


theorem AffineMap.lineMap_mem {k V P : Type*} [Ring k] [AddCommGroup V] [Module k V]
    [AddTorsor V P] {Q : AffineSubspace k P} {p₀ p₁ : P} (c : k) (h₀ : p₀ ∈ Q) (h₁ : p₁ ∈ Q) :
    AffineMap.lineMap p₀ p₁ c ∈ Q := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    Q : AffineSubspace k P
    p₀ p₁ : P
    c : k
    h₀ : Membership.mem Q p₀
    h₁ : Membership.mem Q p₁
    ⊢ Membership.mem Q ((AffineMap.lineMap p₀ p₁) c)
  -/
  rw [AffineMap.lineMap_apply]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    Q : AffineSubspace k P
    p₀ p₁ : P
    c : k
    h₀ : Membership.mem Q p₀
    h₁ : Membership.mem Q p₁
    ⊢ Membership.mem Q (HVAdd.hVAdd (HSMul.hSMul c (VSub.vsub p₁ p₀)) p₀)
  -/
  exact Q.smul_vsub_vadd_mem c h₁ h₀ h₀
  /-
    🎉 no goals
  -/


/-- The affine span of a set of points is the smallest affine subspace containing those points.
(Actually defined here in terms of spans in modules.) -/
def affineSpan (s : Set P) : AffineSubspace k P where
  carrier := spanPoints k s
  smul_vsub_vadd_mem c _ _ _ hp1 hp2 hp3 :=
    vadd_mem_spanPoints_of_mem_spanPoints_of_mem_vectorSpan k hp3
      ((vectorSpan k s).smul_mem c
        (vsub_mem_vectorSpan_of_mem_spanPoints_of_mem_spanPoints k hp1 hp2))


/-- The affine span, converted to a set, is `spanPoints`. -/
@[simp]
theorem coe_affineSpan (s : Set P) : (affineSpan k s : Set P) = spanPoints k s :=
  rfl


/-- A set is contained in its affine span. -/
theorem subset_affineSpan (s : Set P) : s ⊆ affineSpan k s :=
  subset_spanPoints k s


/-- The direction of the affine span is the `vectorSpan`. -/
theorem direction_affineSpan (s : Set P) : (affineSpan k s).direction = vectorSpan k s := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    ⊢ Eq (affineSpan k s).direction (vectorSpan k s)
  -/
  apply le_antisymm
    /-
      case a
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      ⊢ LE.le (affineSpan k s).direction (vectorSpan k s)
    -/
  · refine Submodule.span_le.2 ?_
    /-
      case a
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      ⊢ HasSubset.Subset (VSub.vsub ↑(affineSpan k s) ↑(affineSpan k s)) ↑(vectorSpa …
    -/
    rintro v ⟨p1, ⟨p2, hp2, v1, hv1, hp1⟩, p3, ⟨p4, hp4, v2, hv2, hp3⟩, rfl⟩
    /-
      case a.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p1 p2 : P
      hp2 : Membership.mem s p2
      v1 : V
      hv1 : Membership.mem (vectorSpan k s) v1
      hp1 : Eq p1 (HVAdd.hVAdd v1 p2)
      p3 p4 : P
      hp4 : Membership.mem s p4
      v2 : V
      hv2 : Membership.mem (vectorSpan k s) v2
      hp3 : Eq p3 (HVAdd.hVAdd v2 p4)
      ⊢ Membership.mem (↑(vectorSpan k s)) ((fun x1 x2 => VSub.vsub x1 x2) p1 p3)
    -/
    simp only [SetLike.mem_coe]
    /-
      case a.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p1 p2 : P
      hp2 : Membership.mem s p2
      v1 : V
      hv1 : Membership.mem (vectorSpan k s) v1
      hp1 : Eq p1 (HVAdd.hVAdd v1 p2)
      p3 p4 : P
      hp4 : Membership.mem s p4
      v2 : V
      hv2 : Membership.mem (vectorSpan k s) v2
      hp3 : Eq p3 (HVAdd.hVAdd v2 p4)
      ⊢ Membership.mem (vectorSpan k s) (VSub.vsub p1 p3)
    -/
    rw [hp1, hp3, vsub_vadd_eq_vsub_sub, vadd_vsub_assoc]
    exact
      (vectorSpan k s).sub_mem ((vectorSpan k s).add_mem hv1 (vsub_mem_vectorSpan k hp2 hp4)) hv2
    /-
      case a
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      ⊢ LE.le (vectorSpan k s) (affineSpan k s).direction
    -/
  · exact vectorSpan_mono k (subset_spanPoints k s)
    /-
      🎉 no goals
    -/


/-- A point in a set is in its affine span. -/
theorem mem_affineSpan {p : P} {s : Set P} (hp : p ∈ s) : p ∈ affineSpan k s :=
  mem_spanPoints k p s hp


@[simp]
lemma vectorSpan_add_self (s : Set V) : (vectorSpan k s : Set V) + s = affineSpan k s := by
  /-
    k : Type u_1
    V : Type u_2
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s : Set V
    ⊢ Eq (HAdd.hAdd (↑(vectorSpan k s)) s) ↑(affineSpan k s)
  -/
  ext
  /-
    case h
    k : Type u_1
    V : Type u_2
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s : Set V
    x✝ : V
    ⊢ Iff (Membership.mem (HAdd.hAdd (↑(vectorSpan k s)) s) x✝) (Membership.mem (↑ …
  -/
  simp [mem_add, spanPoints]
  /-
    case h
    k : Type u_1
    V : Type u_2
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s : Set V
    x✝ : V
    ⊢ Iff (Exists fun x => And (Membership.mem (vectorSpan k s) x) (Exists fun y = …
  -/
  aesop
  /-
    🎉 no goals
  -/


instance : CompleteLattice (AffineSubspace k P) :=
  {
    PartialOrder.lift ((↑) : AffineSubspace k P → Set P)
      coe_injective with
    sup := fun s1 s2 => affineSpan k (s1 ∪ s2)
    le_sup_left := fun _ _ =>
      Set.Subset.trans Set.subset_union_left (subset_spanPoints k _)
    le_sup_right := fun _ _ =>
      Set.Subset.trans Set.subset_union_right (subset_spanPoints k _)
    sup_le := fun _ _ _ hs1 hs2 => spanPoints_subset_coe_of_subset_coe (Set.union_subset hs1 hs2)
    inf := fun s1 s2 =>
      mk (s1 ∩ s2) fun c _ _ _ hp1 hp2 hp3 =>
        ⟨s1.smul_vsub_vadd_mem c hp1.1 hp2.1 hp3.1, s2.smul_vsub_vadd_mem c hp1.2 hp2.2 hp3.2⟩
    inf_le_left := fun _ _ => Set.inter_subset_left
    inf_le_right := fun _ _ => Set.inter_subset_right
    le_sInf := fun S s1 hs1 => by
      -- Porting note: surely there is an easier way?
      /-
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝² : Ring k
        inst✝¹ : AddCommGroup V
        inst✝ : Module k V
        S✝ : AddTorsor V P
        S : Set (AffineSubspace k P)
        s1 : AffineSubspace k P
        hs1 : ∀ (b : AffineSubspace k P), Membership.mem S b → LE.le s1 b
        ⊢ LE.le s1 (InfSet.sInf S)
      -/
      refine Set.subset_sInter (t := (s1 : Set P)) ?_
      /-
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝² : Ring k
        inst✝¹ : AddCommGroup V
        inst✝ : Module k V
        S✝ : AddTorsor V P
        S : Set (AffineSubspace k P)
        s1 : AffineSubspace k P
        hs1 : ∀ (b : AffineSubspace k P), Membership.mem S b → LE.le s1 b
        ⊢ ∀ (t' : Set P), Membership.mem (Set.range fun s' => Set.iInter fun h => ↑s') …
      -/
      rintro t ⟨s, _hs, rfl⟩
      /-
        case intro.refl
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝² : Ring k
        inst✝¹ : AddCommGroup V
        inst✝ : Module k V
        S✝ : AddTorsor V P
        S : Set (AffineSubspace k P)
        s1 : AffineSubspace k P
        hs1 : ∀ (b : AffineSubspace k P), Membership.mem S b → LE.le s1 b
        s : AffineSubspace k P
        ⊢ HasSubset.Subset (↑s1) ((fun s' => Set.iInter fun h => ↑s') s)
      -/
      exact Set.subset_iInter (hs1 s)
      /-
        🎉 no goals
      -/
    top :=
      { carrier := Set.univ
        smul_vsub_vadd_mem := fun _ _ _ _ _ _ _ => Set.mem_univ _ }
    le_top := fun _ _ _ => Set.mem_univ _
    bot :=
      { carrier := ∅
        smul_vsub_vadd_mem := fun _ _ _ _ => False.elim }
    bot_le := fun _ _ => False.elim
          /-
            k : Type u_1
            V : Type u_2
            P : Type u_3
            inst✝² : Ring k
            inst✝¹ : AddCommGroup V
            inst✝ : Module k V
            S : AddTorsor V P
            s : Set (AffineSubspace k P)
            c : k
            p1 p2 p3 : P
            hp1 : Membership.mem (Set.iInter fun s' => Set.iInter fun h => ↑s') p1
            hp2 : Membership.mem (Set.iInter fun s' => Set.iInter fun h => ↑s') p2
            hp3 : Membership.mem (Set.iInter fun s' => Set.iInter fun h => ↑s') p3
            s2 : AffineSubspace k P
            hs2 : Membership.mem s s2
            ⊢ Membership.mem (↑s2) (HVAdd.hVAdd (HSMul.hSMul c (VSub.vsub p1 p2)) p3)
          -/
    sSup := fun s => affineSpan k (⋃ s' ∈ s, (s' : Set P))
          /-
            k : Type u_1
            V : Type u_2
            P : Type u_3
            inst✝² : Ring k
            inst✝¹ : AddCommGroup V
            inst✝ : Module k V
            S : AddTorsor V P
            s : Set (AffineSubspace k P)
            c : k
            p1 p2 p3 : P
            hp1 : ∀ (i : AffineSubspace k P), Membership.mem s i → Membership.mem (↑i) p1
            hp2 : ∀ (i : AffineSubspace k P), Membership.mem s i → Membership.mem (↑i) p2
            hp3 : ∀ (i : AffineSubspace k P), Membership.mem s i → Membership.mem (↑i) p3
            s2 : AffineSubspace k P
            hs2 : Membership.mem s s2
            ⊢ Membership.mem (↑s2) (HVAdd.hVAdd (HSMul.hSMul c (VSub.vsub p1 p2)) p3)
          -/
    sInf := fun s =>
          /-
            🎉 no goals
          -/
      mk (⋂ s' ∈ s, (s' : Set P)) fun c p1 p2 p3 hp1 hp2 hp3 =>
        Set.mem_iInter₂.2 fun s2 hs2 => by
          rw [Set.mem_iInter₂] at *
          exact s2.smul_vsub_vadd_mem c (hp1 s2 hs2) (hp2 s2 hs2) (hp3 s2 hs2)
    le_sSup := fun _ _ h => Set.Subset.trans (Set.subset_biUnion_of_mem h) (subset_spanPoints k _)
    sSup_le := fun _ _ h => spanPoints_subset_coe_of_subset_coe (Set.iUnion₂_subset h)
    sInf_le := fun _ _ => Set.biInter_subset_of_mem
    le_inf := fun _ _ _ => Set.subset_inter }


instance : Inhabited (AffineSubspace k P) :=
  ⟨⊤⟩


/-- The `≤` order on subspaces is the same as that on the corresponding sets. -/
theorem le_def (s1 s2 : AffineSubspace k P) : s1 ≤ s2 ↔ (s1 : Set P) ⊆ s2 :=
  Iff.rfl


/-- One subspace is less than or equal to another if and only if all its points are in the second
subspace. -/
theorem le_def' (s1 s2 : AffineSubspace k P) : s1 ≤ s2 ↔ ∀ p ∈ s1, p ∈ s2 :=
  Iff.rfl


/-- The `<` order on subspaces is the same as that on the corresponding sets. -/
theorem lt_def (s1 s2 : AffineSubspace k P) : s1 < s2 ↔ (s1 : Set P) ⊂ s2 :=
  Iff.rfl


/-- One subspace is not less than or equal to another if and only if it has a point not in the
second subspace. -/
theorem not_le_iff_exists (s1 s2 : AffineSubspace k P) : ¬s1 ≤ s2 ↔ ∃ p ∈ s1, p ∉ s2 :=
  Set.not_subset


/-- If a subspace is less than another, there is a point only in the second. -/
theorem exists_of_lt {s1 s2 : AffineSubspace k P} (h : s1 < s2) : ∃ p ∈ s2, p ∉ s1 :=
  Set.exists_of_ssubset h


/-- A subspace is less than another if and only if it is less than or equal to the second subspace
and there is a point only in the second. -/
theorem lt_iff_le_and_exists (s1 s2 : AffineSubspace k P) :
    s1 < s2 ↔ s1 ≤ s2 ∧ ∃ p ∈ s2, p ∉ s1 := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    ⊢ Iff (LT.lt s1 s2) (And (LE.le s1 s2) (Exists fun p => And (Membership.mem s2 …
  -/
  rw [lt_iff_le_not_le, not_le_iff_exists]
  /-
    🎉 no goals
  -/


/-- If an affine subspace is nonempty and contained in another with the same direction, they are
equal. -/
theorem eq_of_direction_eq_of_nonempty_of_le {s₁ s₂ : AffineSubspace k P}
    (hd : s₁.direction = s₂.direction) (hn : (s₁ : Set P).Nonempty) (hle : s₁ ≤ s₂) : s₁ = s₂ :=
  let ⟨p, hp⟩ := hn
  ext_of_direction_eq hd ⟨p, hp, hle hp⟩


/-- The affine span is the `sInf` of subspaces containing the given points. -/
theorem affineSpan_eq_sInf (s : Set P) :
    affineSpan k s = sInf { s' : AffineSubspace k P | s ⊆ s' } :=
  le_antisymm (spanPoints_subset_coe_of_subset_coe <| Set.subset_iInter₂ fun _ => id)
    (sInf_le (subset_spanPoints k _))


/-- The Galois insertion formed by `affineSpan` and coercion back to a set. -/
protected def gi : GaloisInsertion (affineSpan k) ((↑) : AffineSubspace k P → Set P) where
  choice s _ := affineSpan k s
  gc s1 _s2 :=
    ⟨fun h => Set.Subset.trans (subset_spanPoints k s1) h, spanPoints_subset_coe_of_subset_coe⟩
  le_l_u _ := subset_spanPoints k _
  choice_eq _ _ := rfl


/-- The span of the empty set is `⊥`. -/
@[simp]
theorem span_empty : affineSpan k (∅ : Set P) = ⊥ :=
  (AffineSubspace.gi k V P).gc.l_bot


/-- The span of `univ` is `⊤`. -/
@[simp]
theorem span_univ : affineSpan k (Set.univ : Set P) = ⊤ :=
  eq_top_iff.2 <| subset_spanPoints k _


theorem _root_.affineSpan_le {s : Set P} {Q : AffineSubspace k P} :
    affineSpan k s ≤ Q ↔ s ⊆ (Q : Set P) :=
  (AffineSubspace.gi k V P).gc _ _


/-- The affine span of a single point, coerced to a set, contains just that point. -/
@[simp 1001] -- Porting note: this needs to take priority over `coe_affineSpan`
theorem coe_affineSpan_singleton (p : P) : (affineSpan k ({p} : Set P) : Set P) = {p} := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    p : P
    ⊢ Eq (↑(affineSpan k (Singleton.singleton p))) (Singleton.singleton p)
  -/
  ext x
  rw [mem_coe, ← vsub_right_mem_direction_iff_mem (mem_affineSpan k (Set.mem_singleton p)) _,
    direction_affineSpan]
  /-
    case h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    p x : P
    ⊢ Iff (Membership.mem (vectorSpan k (Singleton.singleton p)) (VSub.vsub x p))  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A point is in the affine span of a single point if and only if they are equal. -/
@[simp]
theorem mem_affineSpan_singleton : p₁ ∈ affineSpan k ({p₂} : Set P) ↔ p₁ = p₂ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    p₁ p₂ : P
    ⊢ Iff (Membership.mem (affineSpan k (Singleton.singleton p₂)) p₁) (Eq p₁ p₂)
  -/
  simp [← mem_coe]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_coe_affineSpan_singleton (x : P) :
    ((↑) : affineSpan k ({x} : Set P) → P) ⁻¹' {x} = univ :=
  eq_univ_of_forall fun y => (AffineSubspace.mem_affineSpan_singleton _ _).1 y.2


/-- The span of a union of sets is the sup of their spans. -/
theorem span_union (s t : Set P) : affineSpan k (s ∪ t) = affineSpan k s ⊔ affineSpan k t :=
  (AffineSubspace.gi k V P).gc.l_sup


/-- The span of a union of an indexed family of sets is the sup of their spans. -/
theorem span_iUnion {ι : Type*} (s : ι → Set P) :
    affineSpan k (⋃ i, s i) = ⨆ i, affineSpan k (s i) :=
  (AffineSubspace.gi k V P).gc.l_iSup


/-- `⊤`, coerced to a set, is the whole set of points. -/
@[simp]
theorem top_coe : ((⊤ : AffineSubspace k P) : Set P) = Set.univ :=
  rfl


/-- All points are in `⊤`. -/
@[simp]
theorem mem_top (p : P) : p ∈ (⊤ : AffineSubspace k P) :=
  Set.mem_univ p


/-- The direction of `⊤` is the whole module as a submodule. -/
@[simp]
theorem direction_top : (⊤ : AffineSubspace k P).direction = ⊤ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ⊢ Eq Top.top.direction Top.top
  -/
  cases' S.nonempty with p
  /-
    case intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    p : P
    ⊢ Eq Top.top.direction Top.top
  -/
  ext v
  /-
    case intro.h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    p : P
    v : V
    ⊢ Iff (Membership.mem Top.top.direction v) (Membership.mem Top.top v)
  -/
  refine ⟨imp_intro Submodule.mem_top, fun _hv => ?_⟩
  have hpv : ((v +ᵥ p) -ᵥ p : V) ∈ (⊤ : AffineSubspace k P).direction :=
    vsub_mem_direction (mem_top k V _) (mem_top k V _)
  /-
    case intro.h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    p : P
    v : V
    _hv : Membership.mem Top.top v
    hpv : Membership.mem Top.top.direction (VSub.vsub (HVAdd.hVAdd v p) p)
    ⊢ Membership.mem Top.top.direction v
  -/
  rwa [vadd_vsub] at hpv
  /-
    🎉 no goals
  -/


/-- `⊥`, coerced to a set, is the empty set. -/
@[simp]
theorem bot_coe : ((⊥ : AffineSubspace k P) : Set P) = ∅ :=
  rfl


theorem bot_ne_top : (⊥ : AffineSubspace k P) ≠ ⊤ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ⊢ Ne Bot.bot Top.top
  -/
  intro contra
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    contra : Eq Bot.bot Top.top
    ⊢ False
  -/
  rw [AffineSubspace.ext_iff, bot_coe, top_coe] at contra
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    contra : Eq EmptyCollection.emptyCollection Set.univ
    ⊢ False
  -/
  exact Set.empty_ne_univ contra
  /-
    🎉 no goals
  -/


instance : Nontrivial (AffineSubspace k P) :=
  ⟨⟨⊥, ⊤, bot_ne_top k V P⟩⟩


theorem nonempty_of_affineSpan_eq_top {s : Set P} (h : affineSpan k s = ⊤) : s.Nonempty := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : Set P
    h : Eq (affineSpan k s) Top.top
    ⊢ s.Nonempty
  -/
  rw [Set.nonempty_iff_ne_empty]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : Set P
    h : Eq (affineSpan k s) Top.top
    ⊢ Ne s EmptyCollection.emptyCollection
  -/
  rintro rfl
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    h : Eq (affineSpan k EmptyCollection.emptyCollection) Top.top
    ⊢ False
  -/
  rw [AffineSubspace.span_empty] at h
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    h : Eq Bot.bot Top.top
    ⊢ False
  -/
  exact bot_ne_top k V P h
  /-
    🎉 no goals
  -/


/-- If the affine span of a set is `⊤`, then the vector span of the same set is the `⊤`. -/
theorem vectorSpan_eq_top_of_affineSpan_eq_top {s : Set P} (h : affineSpan k s = ⊤) :
                             /-
                               k : Type u_1
                               V : Type u_2
                               P : Type u_3
                               inst✝² : Ring k
                               inst✝¹ : AddCommGroup V
                               inst✝ : Module k V
                               S : AddTorsor V P
                               s : Set P
                               h : Eq (affineSpan k s) Top.top
                               ⊢ Eq (vectorSpan k s) Top.top
                             -/
    vectorSpan k s = ⊤ := by rw [← direction_affineSpan, h, direction_top]
                             /-
                               🎉 no goals
                             -/


/-- For a nonempty set, the affine span is `⊤` iff its vector span is `⊤`. -/
theorem affineSpan_eq_top_iff_vectorSpan_eq_top_of_nonempty {s : Set P} (hs : s.Nonempty) :
    affineSpan k s = ⊤ ↔ vectorSpan k s = ⊤ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : Set P
    hs : s.Nonempty
    ⊢ Iff (Eq (affineSpan k s) Top.top) (Eq (vectorSpan k s) Top.top)
  -/
  refine ⟨vectorSpan_eq_top_of_affineSpan_eq_top k V P, ?_⟩
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : Set P
    hs : s.Nonempty
    ⊢ Eq (vectorSpan k s) Top.top → Eq (affineSpan k s) Top.top
  -/
  intro h
  suffices Nonempty (affineSpan k s) by
    obtain ⟨p, hp : p ∈ affineSpan k s⟩ := this
    rw [eq_iff_direction_eq_of_mem hp (mem_top k V p), direction_affineSpan, h, direction_top]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : Set P
    hs : s.Nonempty
    h : Eq (vectorSpan k s) Top.top
    ⊢ Nonempty (Subtype fun x => Membership.mem (affineSpan k s) x)
  -/
  obtain ⟨x, hx⟩ := hs
  /-
    case intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : Set P
    h : Eq (vectorSpan k s) Top.top
    x : P
    hx : Membership.mem s x
    ⊢ Nonempty (Subtype fun x => Membership.mem (affineSpan k s) x)
  -/
  exact ⟨⟨x, mem_affineSpan k hx⟩⟩
  /-
    🎉 no goals
  -/


/-- For a non-trivial space, the affine span of a set is `⊤` iff its vector span is `⊤`. -/
theorem affineSpan_eq_top_iff_vectorSpan_eq_top_of_nontrivial {s : Set P} [Nontrivial P] :
    affineSpan k s = ⊤ ↔ vectorSpan k s = ⊤ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    s : Set P
    inst✝ : Nontrivial P
    ⊢ Iff (Eq (affineSpan k s) Top.top) (Eq (vectorSpan k s) Top.top)
  -/
  rcases s.eq_empty_or_nonempty with hs | hs
    /-
      case inl
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      S : AddTorsor V P
      s : Set P
      inst✝ : Nontrivial P
      hs : Eq s EmptyCollection.emptyCollection
      ⊢ Iff (Eq (affineSpan k s) Top.top) (Eq (vectorSpan k s) Top.top)
    -/
  · simp [hs, subsingleton_iff_bot_eq_top, AddTorsor.subsingleton_iff V P, not_subsingleton]
    /-
      🎉 no goals
    -/
    /-
      case inr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      S : AddTorsor V P
      s : Set P
      inst✝ : Nontrivial P
      hs : s.Nonempty
      ⊢ Iff (Eq (affineSpan k s) Top.top) (Eq (vectorSpan k s) Top.top)
    -/
  · rw [affineSpan_eq_top_iff_vectorSpan_eq_top_of_nonempty k V P hs]
    /-
      🎉 no goals
    -/


theorem card_pos_of_affineSpan_eq_top {ι : Type*} [Fintype ι] {p : ι → P}
    (h : affineSpan k (range p) = ⊤) : 0 < Fintype.card ι := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    inst✝ : Fintype ι
    p : ι → P
    h : Eq (affineSpan k (Set.range p)) Top.top
    ⊢ LT.lt 0 (Fintype.card ι)
  -/
  obtain ⟨-, ⟨i, -⟩⟩ := nonempty_of_affineSpan_eq_top k V P h
  /-
    case intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    S : AddTorsor V P
    ι : Type u_4
    inst✝ : Fintype ι
    p : ι → P
    h : Eq (affineSpan k (Set.range p)) Top.top
    i : ι
    ⊢ LT.lt 0 (Fintype.card ι)
  -/
  exact Fintype.card_pos_iff.mpr ⟨i⟩
  /-
    🎉 no goals
  -/


instance : Nonempty (⊤ : AffineSubspace k P) := inferInstanceAs (Nonempty (⊤ : Set P))


/-- The top affine subspace is linearly equivalent to the affine space.
This is the affine version of `Submodule.topEquiv`. -/
@[simps! linear apply symm_apply_coe]
def topEquiv : (⊤ : AffineSubspace k P) ≃ᵃ[k] P where
  toEquiv := Equiv.Set.univ P
  linear := .ofEq _ _ (direction_top _ _ _) ≪≫ₗ Submodule.topEquiv
  map_vadd' _p _v := rfl


/-- No points are in `⊥`. -/
theorem not_mem_bot (p : P) : p ∉ (⊥ : AffineSubspace k P) :=
  Set.not_mem_empty p


/-- The direction of `⊥` is the submodule `⊥`. -/
@[simp]
theorem direction_bot : (⊥ : AffineSubspace k P).direction = ⊥ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    ⊢ Eq Bot.bot.direction Bot.bot
  -/
  rw [direction_eq_vectorSpan, bot_coe, vectorSpan_def, vsub_empty, Submodule.span_empty]
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_eq_bot_iff (Q : AffineSubspace k P) : (Q : Set P) = ∅ ↔ Q = ⊥ :=
  coe_injective.eq_iff' (bot_coe _ _ _)


@[simp]
theorem coe_eq_univ_iff (Q : AffineSubspace k P) : (Q : Set P) = univ ↔ Q = ⊤ :=
  coe_injective.eq_iff' (top_coe _ _ _)


theorem nonempty_iff_ne_bot (Q : AffineSubspace k P) : (Q : Set P).Nonempty ↔ Q ≠ ⊥ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    Q : AffineSubspace k P
    ⊢ Iff (↑Q).Nonempty (Ne Q Bot.bot)
  -/
  rw [nonempty_iff_ne_empty]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    Q : AffineSubspace k P
    ⊢ Iff (Ne (↑Q) EmptyCollection.emptyCollection) (Ne Q Bot.bot)
  -/
  exact not_congr Q.coe_eq_bot_iff
  /-
    🎉 no goals
  -/


theorem eq_bot_or_nonempty (Q : AffineSubspace k P) : Q = ⊥ ∨ (Q : Set P).Nonempty := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    Q : AffineSubspace k P
    ⊢ Or (Eq Q Bot.bot) (↑Q).Nonempty
  -/
  rw [nonempty_iff_ne_bot]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    Q : AffineSubspace k P
    ⊢ Or (Eq Q Bot.bot) (Ne Q Bot.bot)
  -/
  apply eq_or_ne
  /-
    🎉 no goals
  -/


theorem subsingleton_of_subsingleton_span_eq_top {s : Set P} (h₁ : s.Subsingleton)
    (h₂ : affineSpan k s = ⊤) : Subsingleton P := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : Set P
    h₁ : s.Subsingleton
    h₂ : Eq (affineSpan k s) Top.top
    ⊢ Subsingleton P
  -/
  obtain ⟨p, hp⟩ := AffineSubspace.nonempty_of_affineSpan_eq_top k V P h₂
  /-
    case intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : Set P
    h₁ : s.Subsingleton
    h₂ : Eq (affineSpan k s) Top.top
    p : P
    hp : Membership.mem s p
    ⊢ Subsingleton P
  -/
  have : s = {p} := Subset.antisymm (fun q hq => h₁ hq hp) (by simp [hp])
  rw [this, AffineSubspace.ext_iff, AffineSubspace.coe_affineSpan_singleton,
    AffineSubspace.top_coe, eq_comm, ← subsingleton_iff_singleton (mem_univ _)] at h₂
  /-
    case intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : Set P
    h₁ : s.Subsingleton
    p : P
    h₂ : Set.univ.Subsingleton
    hp : Membership.mem s p
    this : Eq s (Singleton.singleton p)
    ⊢ Subsingleton P
  -/
  exact subsingleton_of_univ_subsingleton h₂
  /-
    🎉 no goals
  -/


theorem eq_univ_of_subsingleton_span_eq_top {s : Set P} (h₁ : s.Subsingleton)
    (h₂ : affineSpan k s = ⊤) : s = (univ : Set P) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : Set P
    h₁ : s.Subsingleton
    h₂ : Eq (affineSpan k s) Top.top
    ⊢ Eq s Set.univ
  -/
  obtain ⟨p, hp⟩ := AffineSubspace.nonempty_of_affineSpan_eq_top k V P h₂
  /-
    case intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : Set P
    h₁ : s.Subsingleton
    h₂ : Eq (affineSpan k s) Top.top
    p : P
    hp : Membership.mem s p
    ⊢ Eq s Set.univ
  -/
  have : s = {p} := Subset.antisymm (fun q hq => h₁ hq hp) (by simp [hp])
  /-
    case intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : Set P
    h₁ : s.Subsingleton
    h₂ : Eq (affineSpan k s) Top.top
    p : P
    hp : Membership.mem s p
    this : Eq s (Singleton.singleton p)
    ⊢ Eq s Set.univ
  -/
  rw [this, eq_comm, ← subsingleton_iff_singleton (mem_univ p), subsingleton_univ_iff]
  /-
    case intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : Set P
    h₁ : s.Subsingleton
    h₂ : Eq (affineSpan k s) Top.top
    p : P
    hp : Membership.mem s p
    this : Eq s (Singleton.singleton p)
    ⊢ Subsingleton P
  -/
  exact subsingleton_of_subsingleton_span_eq_top h₁ h₂
  /-
    🎉 no goals
  -/


/-- A nonempty affine subspace is `⊤` if and only if its direction is `⊤`. -/
@[simp]
theorem direction_eq_top_iff_of_nonempty {s : AffineSubspace k P} (h : (s : Set P).Nonempty) :
    s.direction = ⊤ ↔ s = ⊤ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : AffineSubspace k P
    h : (↑s).Nonempty
    ⊢ Iff (Eq s.direction Top.top) (Eq s Top.top)
  -/
  constructor
    /-
      case mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝² : Ring k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      S : AddTorsor V P
      s : AffineSubspace k P
      h : (↑s).Nonempty
      ⊢ Eq s.direction Top.top → Eq s Top.top
    -/
  · intro hd
    /-
      case mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝² : Ring k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      S : AddTorsor V P
      s : AffineSubspace k P
      h : (↑s).Nonempty
      hd : Eq s.direction Top.top
      ⊢ Eq s Top.top
    -/
    rw [← direction_top k V P] at hd
    /-
      case mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝² : Ring k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      S : AddTorsor V P
      s : AffineSubspace k P
      h : (↑s).Nonempty
      hd : Eq s.direction Top.top.direction
      ⊢ Eq s Top.top
    -/
    refine ext_of_direction_eq hd ?_
    /-
      case mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝² : Ring k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      S : AddTorsor V P
      s : AffineSubspace k P
      h : (↑s).Nonempty
      hd : Eq s.direction Top.top.direction
      ⊢ (Inter.inter ↑s ↑Top.top).Nonempty
    -/
    simp [h]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝² : Ring k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      S : AddTorsor V P
      s : AffineSubspace k P
      h : (↑s).Nonempty
      ⊢ Eq s Top.top → Eq s.direction Top.top
    -/
  · rintro rfl
    /-
      case mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝² : Ring k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      S : AddTorsor V P
      h : (↑Top.top).Nonempty
      ⊢ Eq Top.top.direction Top.top
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The inf of two affine subspaces, coerced to a set, is the intersection of the two sets of
points. -/
@[simp]
theorem inf_coe (s1 s2 : AffineSubspace k P) : (s1 ⊓ s2 : Set P) = (s1 : Set P) ∩ s2 :=
  rfl


/-- A point is in the inf of two affine subspaces if and only if it is in both of them. -/
theorem mem_inf_iff (p : P) (s1 s2 : AffineSubspace k P) : p ∈ s1 ⊓ s2 ↔ p ∈ s1 ∧ p ∈ s2 :=
  Iff.rfl


/-- The direction of the inf of two affine subspaces is less than or equal to the inf of their
directions. -/
theorem direction_inf (s1 s2 : AffineSubspace k P) :
    (s1 ⊓ s2).direction ≤ s1.direction ⊓ s2.direction := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    ⊢ LE.le (Min.min s1 s2).direction (Min.min s1.direction s2.direction)
  -/
  simp only [direction_eq_vectorSpan, vectorSpan_def]
  exact
    le_inf (sInf_le_sInf fun p hp => trans (vsub_self_mono inter_subset_left) hp)
      (sInf_le_sInf fun p hp => trans (vsub_self_mono inter_subset_right) hp)


/-- If two affine subspaces have a point in common, the direction of their inf equals the inf of
their directions. -/
theorem direction_inf_of_mem {s₁ s₂ : AffineSubspace k P} {p : P} (h₁ : p ∈ s₁) (h₂ : p ∈ s₂) :
    (s₁ ⊓ s₂).direction = s₁.direction ⊓ s₂.direction := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s₁ s₂ : AffineSubspace k P
    p : P
    h₁ : Membership.mem s₁ p
    h₂ : Membership.mem s₂ p
    ⊢ Eq (Min.min s₁ s₂).direction (Min.min s₁.direction s₂.direction)
  -/
  ext v
  rw [Submodule.mem_inf, ← vadd_mem_iff_mem_direction v h₁, ← vadd_mem_iff_mem_direction v h₂, ←
    vadd_mem_iff_mem_direction v ((mem_inf_iff p s₁ s₂).2 ⟨h₁, h₂⟩), mem_inf_iff]


/-- If two affine subspaces have a point in their inf, the direction of their inf equals the inf of
their directions. -/
theorem direction_inf_of_mem_inf {s₁ s₂ : AffineSubspace k P} {p : P} (h : p ∈ s₁ ⊓ s₂) :
    (s₁ ⊓ s₂).direction = s₁.direction ⊓ s₂.direction :=
  direction_inf_of_mem ((mem_inf_iff p s₁ s₂).1 h).1 ((mem_inf_iff p s₁ s₂).1 h).2


/-- If one affine subspace is less than or equal to another, the same applies to their
directions. -/
theorem direction_le {s1 s2 : AffineSubspace k P} (h : s1 ≤ s2) : s1.direction ≤ s2.direction := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h : LE.le s1 s2
    ⊢ LE.le s1.direction s2.direction
  -/
  simp only [direction_eq_vectorSpan, vectorSpan_def]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h : LE.le s1 s2
    ⊢ LE.le (Submodule.span k (VSub.vsub ↑s1 ↑s1)) (Submodule.span k (VSub.vsub ↑s …
  -/
  exact vectorSpan_mono k h
  /-
    🎉 no goals
  -/


/-- If one nonempty affine subspace is less than another, the same applies to their directions -/
theorem direction_lt_of_nonempty {s1 s2 : AffineSubspace k P} (h : s1 < s2)
    (hn : (s1 : Set P).Nonempty) : s1.direction < s2.direction := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h : LT.lt s1 s2
    hn : (↑s1).Nonempty
    ⊢ LT.lt s1.direction s2.direction
  -/
  cases' hn with p hp
  /-
    case intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h : LT.lt s1 s2
    p : P
    hp : Membership.mem (↑s1) p
    ⊢ LT.lt s1.direction s2.direction
  -/
  rw [lt_iff_le_and_exists] at h
  /-
    case intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h : And (LE.le s1 s2) (Exists fun p => And (Membership.mem s2 p) (Not (Members …
    p : P
    hp : Membership.mem (↑s1) p
    ⊢ LT.lt s1.direction s2.direction
  -/
  rcases h with ⟨hle, p2, hp2, hp2s1⟩
  /-
    case intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    p : P
    hp : Membership.mem (↑s1) p
    hle : LE.le s1 s2
    p2 : P
    hp2 : Membership.mem s2 p2
    hp2s1 : Not (Membership.mem s1 p2)
    ⊢ LT.lt s1.direction s2.direction
  -/
  rw [SetLike.lt_iff_le_and_exists]
  /-
    case intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    p : P
    hp : Membership.mem (↑s1) p
    hle : LE.le s1 s2
    p2 : P
    hp2 : Membership.mem s2 p2
    hp2s1 : Not (Membership.mem s1 p2)
    ⊢ And (LE.le s1.direction s2.direction) (Exists fun x => And (Membership.mem s …
  -/
  use direction_le hle, p2 -ᵥ p, vsub_mem_direction hp2 (hle hp)
  /-
    case right
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    p : P
    hp : Membership.mem (↑s1) p
    hle : LE.le s1 s2
    p2 : P
    hp2 : Membership.mem s2 p2
    hp2s1 : Not (Membership.mem s1 p2)
    ⊢ Not (Membership.mem s1.direction (VSub.vsub p2 p))
  -/
  intro hm
  /-
    case right
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    p : P
    hp : Membership.mem (↑s1) p
    hle : LE.le s1 s2
    p2 : P
    hp2 : Membership.mem s2 p2
    hp2s1 : Not (Membership.mem s1 p2)
    hm : Membership.mem s1.direction (VSub.vsub p2 p)
    ⊢ False
  -/
  rw [vsub_right_mem_direction_iff_mem hp p2] at hm
  /-
    case right
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    p : P
    hp : Membership.mem (↑s1) p
    hle : LE.le s1 s2
    p2 : P
    hp2 : Membership.mem s2 p2
    hp2s1 : Not (Membership.mem s1 p2)
    hm : Membership.mem s1 p2
    ⊢ False
  -/
  exact hp2s1 hm
  /-
    🎉 no goals
  -/


/-- The sup of the directions of two affine subspaces is less than or equal to the direction of
their sup. -/
theorem sup_direction_le (s1 s2 : AffineSubspace k P) :
    s1.direction ⊔ s2.direction ≤ (s1 ⊔ s2).direction := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    ⊢ LE.le (Max.max s1.direction s2.direction) (Max.max s1 s2).direction
  -/
  simp only [direction_eq_vectorSpan, vectorSpan_def]
  exact
    sup_le
      (sInf_le_sInf fun p hp => Set.Subset.trans (vsub_self_mono (le_sup_left : s1 ≤ s1 ⊔ s2)) hp)
      (sInf_le_sInf fun p hp => Set.Subset.trans (vsub_self_mono (le_sup_right : s2 ≤ s1 ⊔ s2)) hp)


/-- The sup of the directions of two nonempty affine subspaces with empty intersection is less than
the direction of their sup. -/
theorem sup_direction_lt_of_nonempty_of_inter_empty {s1 s2 : AffineSubspace k P}
    (h1 : (s1 : Set P).Nonempty) (h2 : (s2 : Set P).Nonempty) (he : (s1 ∩ s2 : Set P) = ∅) :
    s1.direction ⊔ s2.direction < (s1 ⊔ s2).direction := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h1 : (↑s1).Nonempty
    h2 : (↑s2).Nonempty
    he : Eq (Inter.inter ↑s1 ↑s2) EmptyCollection.emptyCollection
    ⊢ LT.lt (Max.max s1.direction s2.direction) (Max.max s1 s2).direction
  -/
  cases' h1 with p1 hp1
  /-
    case intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h2 : (↑s2).Nonempty
    he : Eq (Inter.inter ↑s1 ↑s2) EmptyCollection.emptyCollection
    p1 : P
    hp1 : Membership.mem (↑s1) p1
    ⊢ LT.lt (Max.max s1.direction s2.direction) (Max.max s1 s2).direction
  -/
  cases' h2 with p2 hp2
  /-
    case intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    he : Eq (Inter.inter ↑s1 ↑s2) EmptyCollection.emptyCollection
    p1 : P
    hp1 : Membership.mem (↑s1) p1
    p2 : P
    hp2 : Membership.mem (↑s2) p2
    ⊢ LT.lt (Max.max s1.direction s2.direction) (Max.max s1 s2).direction
  -/
  rw [SetLike.lt_iff_le_and_exists]
  use sup_direction_le s1 s2, p2 -ᵥ p1,
    vsub_mem_direction ((le_sup_right : s2 ≤ s1 ⊔ s2) hp2) ((le_sup_left : s1 ≤ s1 ⊔ s2) hp1)
  /-
    case right
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    he : Eq (Inter.inter ↑s1 ↑s2) EmptyCollection.emptyCollection
    p1 : P
    hp1 : Membership.mem (↑s1) p1
    p2 : P
    hp2 : Membership.mem (↑s2) p2
    ⊢ Not (Membership.mem (Max.max s1.direction s2.direction) (VSub.vsub p2 p1))
  -/
  intro h
  /-
    case right
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    he : Eq (Inter.inter ↑s1 ↑s2) EmptyCollection.emptyCollection
    p1 : P
    hp1 : Membership.mem (↑s1) p1
    p2 : P
    hp2 : Membership.mem (↑s2) p2
    h : Membership.mem (Max.max s1.direction s2.direction) (VSub.vsub p2 p1)
    ⊢ False
  -/
  rw [Submodule.mem_sup] at h
  /-
    case right
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    he : Eq (Inter.inter ↑s1 ↑s2) EmptyCollection.emptyCollection
    p1 : P
    hp1 : Membership.mem (↑s1) p1
    p2 : P
    hp2 : Membership.mem (↑s2) p2
    h : Exists fun y => And (Membership.mem s1.direction y) (Exists fun z => And ( …
    ⊢ False
  -/
  rcases h with ⟨v1, hv1, v2, hv2, hv1v2⟩
  rw [← sub_eq_zero, sub_eq_add_neg, neg_vsub_eq_vsub_rev, add_comm v1, add_assoc, ←
    vadd_vsub_assoc, ← neg_neg v2, add_comm, ← sub_eq_add_neg, ← vsub_vadd_eq_vsub_sub,
    vsub_eq_zero_iff_eq] at hv1v2
  /-
    case right.intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    he : Eq (Inter.inter ↑s1 ↑s2) EmptyCollection.emptyCollection
    p1 : P
    hp1 : Membership.mem (↑s1) p1
    p2 : P
    hp2 : Membership.mem (↑s2) p2
    v1 : V
    hv1 : Membership.mem s1.direction v1
    v2 : V
    hv2 : Membership.mem s2.direction v2
    hv1v2 : Eq (HVAdd.hVAdd v1 p1) (HVAdd.hVAdd (Neg.neg v2) p2)
    ⊢ False
  -/
  refine Set.Nonempty.ne_empty ?_ he
  /-
    case right.intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    he : Eq (Inter.inter ↑s1 ↑s2) EmptyCollection.emptyCollection
    p1 : P
    hp1 : Membership.mem (↑s1) p1
    p2 : P
    hp2 : Membership.mem (↑s2) p2
    v1 : V
    hv1 : Membership.mem s1.direction v1
    v2 : V
    hv2 : Membership.mem s2.direction v2
    hv1v2 : Eq (HVAdd.hVAdd v1 p1) (HVAdd.hVAdd (Neg.neg v2) p2)
    ⊢ (Inter.inter ↑s1 ↑s2).Nonempty
  -/
  use v1 +ᵥ p1, vadd_mem_of_mem_direction hv1 hp1
  /-
    case right
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    he : Eq (Inter.inter ↑s1 ↑s2) EmptyCollection.emptyCollection
    p1 : P
    hp1 : Membership.mem (↑s1) p1
    p2 : P
    hp2 : Membership.mem (↑s2) p2
    v1 : V
    hv1 : Membership.mem s1.direction v1
    v2 : V
    hv2 : Membership.mem s2.direction v2
    hv1v2 : Eq (HVAdd.hVAdd v1 p1) (HVAdd.hVAdd (Neg.neg v2) p2)
    ⊢ Membership.mem (↑s2) (HVAdd.hVAdd v1 p1)
  -/
  rw [hv1v2]
  /-
    case right
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    he : Eq (Inter.inter ↑s1 ↑s2) EmptyCollection.emptyCollection
    p1 : P
    hp1 : Membership.mem (↑s1) p1
    p2 : P
    hp2 : Membership.mem (↑s2) p2
    v1 : V
    hv1 : Membership.mem s1.direction v1
    v2 : V
    hv2 : Membership.mem s2.direction v2
    hv1v2 : Eq (HVAdd.hVAdd v1 p1) (HVAdd.hVAdd (Neg.neg v2) p2)
    ⊢ Membership.mem (↑s2) (HVAdd.hVAdd (Neg.neg v2) p2)
  -/
  exact vadd_mem_of_mem_direction (Submodule.neg_mem _ hv2) hp2
  /-
    🎉 no goals
  -/


/-- If the directions of two nonempty affine subspaces span the whole module, they have nonempty
intersection. -/
theorem inter_nonempty_of_nonempty_of_sup_direction_eq_top {s1 s2 : AffineSubspace k P}
    (h1 : (s1 : Set P).Nonempty) (h2 : (s2 : Set P).Nonempty)
    (hd : s1.direction ⊔ s2.direction = ⊤) : ((s1 : Set P) ∩ s2).Nonempty := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h1 : (↑s1).Nonempty
    h2 : (↑s2).Nonempty
    hd : Eq (Max.max s1.direction s2.direction) Top.top
    ⊢ (Inter.inter ↑s1 ↑s2).Nonempty
  -/
  by_contra h
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h1 : (↑s1).Nonempty
    h2 : (↑s2).Nonempty
    hd : Eq (Max.max s1.direction s2.direction) Top.top
    h : Not (Inter.inter ↑s1 ↑s2).Nonempty
    ⊢ False
  -/
  rw [Set.not_nonempty_iff_eq_empty] at h
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h1 : (↑s1).Nonempty
    h2 : (↑s2).Nonempty
    hd : Eq (Max.max s1.direction s2.direction) Top.top
    h : Eq (Inter.inter ↑s1 ↑s2) EmptyCollection.emptyCollection
    ⊢ False
  -/
  have hlt := sup_direction_lt_of_nonempty_of_inter_empty h1 h2 h
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h1 : (↑s1).Nonempty
    h2 : (↑s2).Nonempty
    hd : Eq (Max.max s1.direction s2.direction) Top.top
    h : Eq (Inter.inter ↑s1 ↑s2) EmptyCollection.emptyCollection
    hlt : LT.lt (Max.max s1.direction s2.direction) (Max.max s1 s2).direction
    ⊢ False
  -/
  rw [hd] at hlt
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h1 : (↑s1).Nonempty
    h2 : (↑s2).Nonempty
    hd : Eq (Max.max s1.direction s2.direction) Top.top
    h : Eq (Inter.inter ↑s1 ↑s2) EmptyCollection.emptyCollection
    hlt : LT.lt Top.top (Max.max s1 s2).direction
    ⊢ False
  -/
  exact not_top_lt hlt
  /-
    🎉 no goals
  -/


/-- If the directions of two nonempty affine subspaces are complements of each other, they intersect
in exactly one point. -/
theorem inter_eq_singleton_of_nonempty_of_isCompl {s1 s2 : AffineSubspace k P}
    (h1 : (s1 : Set P).Nonempty) (h2 : (s2 : Set P).Nonempty)
    (hd : IsCompl s1.direction s2.direction) : ∃ p, (s1 : Set P) ∩ s2 = {p} := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h1 : (↑s1).Nonempty
    h2 : (↑s2).Nonempty
    hd : IsCompl s1.direction s2.direction
    ⊢ Exists fun p => Eq (Inter.inter ↑s1 ↑s2) (Singleton.singleton p)
  -/
  cases' inter_nonempty_of_nonempty_of_sup_direction_eq_top h1 h2 hd.sup_eq_top with p hp
  /-
    case intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h1 : (↑s1).Nonempty
    h2 : (↑s2).Nonempty
    hd : IsCompl s1.direction s2.direction
    p : P
    hp : Membership.mem (Inter.inter ↑s1 ↑s2) p
    ⊢ Exists fun p => Eq (Inter.inter ↑s1 ↑s2) (Singleton.singleton p)
  -/
  use p
  /-
    case h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h1 : (↑s1).Nonempty
    h2 : (↑s2).Nonempty
    hd : IsCompl s1.direction s2.direction
    p : P
    hp : Membership.mem (Inter.inter ↑s1 ↑s2) p
    ⊢ Eq (Inter.inter ↑s1 ↑s2) (Singleton.singleton p)
  -/
  ext q
  /-
    case h.h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h1 : (↑s1).Nonempty
    h2 : (↑s2).Nonempty
    hd : IsCompl s1.direction s2.direction
    p : P
    hp : Membership.mem (Inter.inter ↑s1 ↑s2) p
    q : P
    ⊢ Iff (Membership.mem (Inter.inter ↑s1 ↑s2) q) (Membership.mem (Singleton.sing …
  -/
  rw [Set.mem_singleton_iff]
  /-
    case h.h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s1 s2 : AffineSubspace k P
    h1 : (↑s1).Nonempty
    h2 : (↑s2).Nonempty
    hd : IsCompl s1.direction s2.direction
    p : P
    hp : Membership.mem (Inter.inter ↑s1 ↑s2) p
    q : P
    ⊢ Iff (Membership.mem (Inter.inter ↑s1 ↑s2) q) (Eq q p)
  -/
  constructor
    /-
      case h.h.mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝² : Ring k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      S : AddTorsor V P
      s1 s2 : AffineSubspace k P
      h1 : (↑s1).Nonempty
      h2 : (↑s2).Nonempty
      hd : IsCompl s1.direction s2.direction
      p : P
      hp : Membership.mem (Inter.inter ↑s1 ↑s2) p
      q : P
      ⊢ Membership.mem (Inter.inter ↑s1 ↑s2) q → Eq q p
    -/
  · rintro ⟨hq1, hq2⟩
    have hqp : q -ᵥ p ∈ s1.direction ⊓ s2.direction :=
      ⟨vsub_mem_direction hq1 hp.1, vsub_mem_direction hq2 hp.2⟩
    /-
      case h.h.mp.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝² : Ring k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      S : AddTorsor V P
      s1 s2 : AffineSubspace k P
      h1 : (↑s1).Nonempty
      h2 : (↑s2).Nonempty
      hd : IsCompl s1.direction s2.direction
      p : P
      hp : Membership.mem (Inter.inter ↑s1 ↑s2) p
      q : P
      hq1 : Membership.mem (↑s1) q
      hq2 : Membership.mem (↑s2) q
      hqp : Membership.mem (Min.min s1.direction s2.direction) (VSub.vsub q p)
      ⊢ Eq q p
    -/
    rwa [hd.inf_eq_bot, Submodule.mem_bot, vsub_eq_zero_iff_eq] at hqp
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝² : Ring k
      inst✝¹ : AddCommGroup V
      inst✝ : Module k V
      S : AddTorsor V P
      s1 s2 : AffineSubspace k P
      h1 : (↑s1).Nonempty
      h2 : (↑s2).Nonempty
      hd : IsCompl s1.direction s2.direction
      p : P
      hp : Membership.mem (Inter.inter ↑s1 ↑s2) p
      q : P
      ⊢ Eq q p → Membership.mem (Inter.inter ↑s1 ↑s2) q
    -/
  · exact fun h => h.symm ▸ hp
    /-
      🎉 no goals
    -/


/-- Coercing a subspace to a set then taking the affine span produces the original subspace. -/
@[simp]
theorem affineSpan_coe (s : AffineSubspace k P) : affineSpan k (s : Set P) = s := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : AffineSubspace k P
    ⊢ Eq (affineSpan k ↑s) s
  -/
  refine le_antisymm ?_ (subset_spanPoints _ _)
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : AffineSubspace k P
    ⊢ LE.le (affineSpan k ↑s) s
  -/
  rintro p ⟨p1, hp1, v, hv, rfl⟩
  /-
    case intro.intro.intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    S : AddTorsor V P
    s : AffineSubspace k P
    p1 : P
    hp1 : Membership.mem (↑s) p1
    v : V
    hv : Membership.mem (vectorSpan k ↑s) v
    ⊢ Membership.mem (↑s) (HVAdd.hVAdd v p1)
  -/
  exact vadd_mem_of_mem_direction hv hp1
  /-
    🎉 no goals
  -/


/-- The `vectorSpan` is the span of the pairwise subtractions with a given point on the left. -/
theorem vectorSpan_eq_span_vsub_set_left {s : Set P} {p : P} (hp : p ∈ s) :
    vectorSpan k s = Submodule.span k ((p -ᵥ ·) '' s) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p : P
    hp : Membership.mem s p
    ⊢ Eq (vectorSpan k s) (Submodule.span k (Set.image (fun x => VSub.vsub p x) s))
  -/
  rw [vectorSpan_def]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p : P
    hp : Membership.mem s p
    ⊢ Eq (Submodule.span k (VSub.vsub s s)) (Submodule.span k (Set.image (fun x => …
  -/
  refine le_antisymm ?_ (Submodule.span_mono ?_)
    /-
      case refine_1
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p : P
      hp : Membership.mem s p
      ⊢ LE.le (Submodule.span k (VSub.vsub s s)) (Submodule.span k (Set.image (fun x …
    -/
  · rw [Submodule.span_le]
    /-
      case refine_1
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p : P
      hp : Membership.mem s p
      ⊢ HasSubset.Subset (VSub.vsub s s) ↑(Submodule.span k (Set.image (fun x => VSu …
    -/
    rintro v ⟨p1, hp1, p2, hp2, hv⟩
    /-
      case refine_1.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p : P
      hp : Membership.mem s p
      v : V
      p1 : P
      hp1 : Membership.mem s p1
      p2 : P
      hp2 : Membership.mem s p2
      hv : Eq ((fun x1 x2 => VSub.vsub x1 x2) p1 p2) v
      ⊢ Membership.mem (↑(Submodule.span k (Set.image (fun x => VSub.vsub p x) s))) v
    -/
    simp_rw [← vsub_sub_vsub_cancel_left p1 p2 p] at hv
    /-
      case refine_1.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p : P
      hp : Membership.mem s p
      v : V
      p1 : P
      hp1 : Membership.mem s p1
      p2 : P
      hp2 : Membership.mem s p2
      hv : Eq (HSub.hSub (VSub.vsub p p2) (VSub.vsub p p1)) v
      ⊢ Membership.mem (↑(Submodule.span k (Set.image (fun x => VSub.vsub p x) s))) v
    -/
    rw [← hv, SetLike.mem_coe, Submodule.mem_span]
    /-
      case refine_1.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p : P
      hp : Membership.mem s p
      v : V
      p1 : P
      hp1 : Membership.mem s p1
      p2 : P
      hp2 : Membership.mem s p2
      hv : Eq (HSub.hSub (VSub.vsub p p2) (VSub.vsub p p1)) v
      ⊢ ∀ (p_1 : Submodule k V), HasSubset.Subset (Set.image (fun x => VSub.vsub p x …
    -/
    exact fun m hm => Submodule.sub_mem _ (hm ⟨p2, hp2, rfl⟩) (hm ⟨p1, hp1, rfl⟩)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p : P
      hp : Membership.mem s p
      ⊢ HasSubset.Subset (Set.image (fun x => VSub.vsub p x) s) (VSub.vsub s s)
    -/
  · rintro v ⟨p2, hp2, hv⟩
    /-
      case refine_2.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p : P
      hp : Membership.mem s p
      v : V
      p2 : P
      hp2 : Membership.mem s p2
      hv : Eq ((fun x => VSub.vsub p x) p2) v
      ⊢ Membership.mem (VSub.vsub s s) v
    -/
    exact ⟨p, hp, p2, hp2, hv⟩
    /-
      🎉 no goals
    -/


/-- The `vectorSpan` is the span of the pairwise subtractions with a given point on the right. -/
theorem vectorSpan_eq_span_vsub_set_right {s : Set P} {p : P} (hp : p ∈ s) :
    vectorSpan k s = Submodule.span k ((· -ᵥ p) '' s) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p : P
    hp : Membership.mem s p
    ⊢ Eq (vectorSpan k s) (Submodule.span k (Set.image (fun x => VSub.vsub x p) s))
  -/
  rw [vectorSpan_def]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p : P
    hp : Membership.mem s p
    ⊢ Eq (Submodule.span k (VSub.vsub s s)) (Submodule.span k (Set.image (fun x => …
  -/
  refine le_antisymm ?_ (Submodule.span_mono ?_)
    /-
      case refine_1
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p : P
      hp : Membership.mem s p
      ⊢ LE.le (Submodule.span k (VSub.vsub s s)) (Submodule.span k (Set.image (fun x …
    -/
  · rw [Submodule.span_le]
    /-
      case refine_1
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p : P
      hp : Membership.mem s p
      ⊢ HasSubset.Subset (VSub.vsub s s) ↑(Submodule.span k (Set.image (fun x => VSu …
    -/
    rintro v ⟨p1, hp1, p2, hp2, hv⟩
    /-
      case refine_1.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p : P
      hp : Membership.mem s p
      v : V
      p1 : P
      hp1 : Membership.mem s p1
      p2 : P
      hp2 : Membership.mem s p2
      hv : Eq ((fun x1 x2 => VSub.vsub x1 x2) p1 p2) v
      ⊢ Membership.mem (↑(Submodule.span k (Set.image (fun x => VSub.vsub x p) s))) v
    -/
    simp_rw [← vsub_sub_vsub_cancel_right p1 p2 p] at hv
    /-
      case refine_1.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p : P
      hp : Membership.mem s p
      v : V
      p1 : P
      hp1 : Membership.mem s p1
      p2 : P
      hp2 : Membership.mem s p2
      hv : Eq (HSub.hSub (VSub.vsub p1 p) (VSub.vsub p2 p)) v
      ⊢ Membership.mem (↑(Submodule.span k (Set.image (fun x => VSub.vsub x p) s))) v
    -/
    rw [← hv, SetLike.mem_coe, Submodule.mem_span]
    /-
      case refine_1.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p : P
      hp : Membership.mem s p
      v : V
      p1 : P
      hp1 : Membership.mem s p1
      p2 : P
      hp2 : Membership.mem s p2
      hv : Eq (HSub.hSub (VSub.vsub p1 p) (VSub.vsub p2 p)) v
      ⊢ ∀ (p_1 : Submodule k V), HasSubset.Subset (Set.image (fun x => VSub.vsub x p …
    -/
    exact fun m hm => Submodule.sub_mem _ (hm ⟨p1, hp1, rfl⟩) (hm ⟨p2, hp2, rfl⟩)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p : P
      hp : Membership.mem s p
      ⊢ HasSubset.Subset (Set.image (fun x => VSub.vsub x p) s) (VSub.vsub s s)
    -/
  · rintro v ⟨p2, hp2, hv⟩
    /-
      case refine_2.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p : P
      hp : Membership.mem s p
      v : V
      p2 : P
      hp2 : Membership.mem s p2
      hv : Eq ((fun x => VSub.vsub x p) p2) v
      ⊢ Membership.mem (VSub.vsub s s) v
    -/
    exact ⟨p2, hp2, p, hp, hv⟩
    /-
      🎉 no goals
    -/


/-- The `vectorSpan` is the span of the pairwise subtractions with a given point on the left,
excluding the subtraction of that point from itself. -/
theorem vectorSpan_eq_span_vsub_set_left_ne {s : Set P} {p : P} (hp : p ∈ s) :
    vectorSpan k s = Submodule.span k ((p -ᵥ ·) '' (s \ {p})) := by
  conv_lhs =>
    rw [vectorSpan_eq_span_vsub_set_left k hp, ← Set.insert_eq_of_mem hp, ←
      Set.insert_diff_singleton, Set.image_insert_eq]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p : P
    hp : Membership.mem s p
    ⊢ Eq (Submodule.span k (Insert.insert (VSub.vsub p p) (Set.image (fun x => VSu …
  -/
  simp [Submodule.span_insert_eq_span]
  /-
    🎉 no goals
  -/


/-- The `vectorSpan` is the span of the pairwise subtractions with a given point on the right,
excluding the subtraction of that point from itself. -/
theorem vectorSpan_eq_span_vsub_set_right_ne {s : Set P} {p : P} (hp : p ∈ s) :
    vectorSpan k s = Submodule.span k ((· -ᵥ p) '' (s \ {p})) := by
  conv_lhs =>
    rw [vectorSpan_eq_span_vsub_set_right k hp, ← Set.insert_eq_of_mem hp, ←
      Set.insert_diff_singleton, Set.image_insert_eq]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p : P
    hp : Membership.mem s p
    ⊢ Eq (Submodule.span k (Insert.insert (VSub.vsub p p) (Set.image (fun x => VSu …
  -/
  simp [Submodule.span_insert_eq_span]
  /-
    🎉 no goals
  -/


/-- The `vectorSpan` is the span of the pairwise subtractions with a given point on the right,
excluding the subtraction of that point from itself. -/
theorem vectorSpan_eq_span_vsub_finset_right_ne [DecidableEq P] [DecidableEq V] {s : Finset P}
    {p : P} (hp : p ∈ s) :
    vectorSpan k (s : Set P) = Submodule.span k ((s.erase p).image (· -ᵥ p)) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁵ : Ring k
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module k V
    inst✝² : AddTorsor V P
    inst✝¹ : DecidableEq P
    inst✝ : DecidableEq V
    s : Finset P
    p : P
    hp : Membership.mem s p
    ⊢ Eq (vectorSpan k ↑s) (Submodule.span k ↑(Finset.image (fun x => VSub.vsub x  …
  -/
  simp [vectorSpan_eq_span_vsub_set_right_ne _ (Finset.mem_coe.mpr hp)]
  /-
    🎉 no goals
  -/


/-- The `vectorSpan` of the image of a function is the span of the pairwise subtractions with a
given point on the left, excluding the subtraction of that point from itself. -/
theorem vectorSpan_image_eq_span_vsub_set_left_ne (p : ι → P) {s : Set ι} {i : ι} (hi : i ∈ s) :
    vectorSpan k (p '' s) = Submodule.span k ((p i -ᵥ ·) '' (p '' (s \ {i}))) := by
  conv_lhs =>
    rw [vectorSpan_eq_span_vsub_set_left k (Set.mem_image_of_mem p hi), ← Set.insert_eq_of_mem hi, ←
      Set.insert_diff_singleton, Set.image_insert_eq, Set.image_insert_eq]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ι : Type u_4
    p : ι → P
    s : Set ι
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (Submodule.span k (Insert.insert (VSub.vsub (p i) (p i)) (Set.image (fun  …
  -/
  simp [Submodule.span_insert_eq_span]
  /-
    🎉 no goals
  -/


/-- The `vectorSpan` of the image of a function is the span of the pairwise subtractions with a
given point on the right, excluding the subtraction of that point from itself. -/
theorem vectorSpan_image_eq_span_vsub_set_right_ne (p : ι → P) {s : Set ι} {i : ι} (hi : i ∈ s) :
    vectorSpan k (p '' s) = Submodule.span k ((· -ᵥ p i) '' (p '' (s \ {i}))) := by
  conv_lhs =>
    rw [vectorSpan_eq_span_vsub_set_right k (Set.mem_image_of_mem p hi), ← Set.insert_eq_of_mem hi,
      ← Set.insert_diff_singleton, Set.image_insert_eq, Set.image_insert_eq]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ι : Type u_4
    p : ι → P
    s : Set ι
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (Submodule.span k (Insert.insert (VSub.vsub (p i) (p i)) (Set.image (fun  …
  -/
  simp [Submodule.span_insert_eq_span]
  /-
    🎉 no goals
  -/


/-- The `vectorSpan` of an indexed family is the span of the pairwise subtractions with a given
point on the left. -/
theorem vectorSpan_range_eq_span_range_vsub_left (p : ι → P) (i0 : ι) :
    vectorSpan k (Set.range p) = Submodule.span k (Set.range fun i : ι => p i0 -ᵥ p i) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ι : Type u_4
    p : ι → P
    i0 : ι
    ⊢ Eq (vectorSpan k (Set.range p)) (Submodule.span k (Set.range fun i => VSub.v …
  -/
  rw [vectorSpan_eq_span_vsub_set_left k (Set.mem_range_self i0), ← Set.range_comp]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ι : Type u_4
    p : ι → P
    i0 : ι
    ⊢ Eq (Submodule.span k (Set.range (Function.comp (fun x => VSub.vsub (p i0) x) …
  -/
  congr
  /-
    🎉 no goals
  -/


/-- The `vectorSpan` of an indexed family is the span of the pairwise subtractions with a given
point on the right. -/
theorem vectorSpan_range_eq_span_range_vsub_right (p : ι → P) (i0 : ι) :
    vectorSpan k (Set.range p) = Submodule.span k (Set.range fun i : ι => p i -ᵥ p i0) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ι : Type u_4
    p : ι → P
    i0 : ι
    ⊢ Eq (vectorSpan k (Set.range p)) (Submodule.span k (Set.range fun i => VSub.v …
  -/
  rw [vectorSpan_eq_span_vsub_set_right k (Set.mem_range_self i0), ← Set.range_comp]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ι : Type u_4
    p : ι → P
    i0 : ι
    ⊢ Eq (Submodule.span k (Set.range (Function.comp (fun x => VSub.vsub x (p i0)) …
  -/
  congr
  /-
    🎉 no goals
  -/


/-- The `vectorSpan` of an indexed family is the span of the pairwise subtractions with a given
point on the left, excluding the subtraction of that point from itself. -/
theorem vectorSpan_range_eq_span_range_vsub_left_ne (p : ι → P) (i₀ : ι) :
    vectorSpan k (Set.range p) =
      Submodule.span k (Set.range fun i : { x // x ≠ i₀ } => p i₀ -ᵥ p i) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ι : Type u_4
    p : ι → P
    i₀ : ι
    ⊢ Eq (vectorSpan k (Set.range p)) (Submodule.span k (Set.range fun i => VSub.v …
  -/
  rw [← Set.image_univ, vectorSpan_image_eq_span_vsub_set_left_ne k _ (Set.mem_univ i₀)]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ι : Type u_4
    p : ι → P
    i₀ : ι
    ⊢ Eq (Submodule.span k (Set.image (fun x => VSub.vsub (p i₀) x) (Set.image p ( …
  -/
  congr with v
  simp only [Set.mem_range, Set.mem_image, Set.mem_diff, Set.mem_singleton_iff, Subtype.exists,
    Subtype.coe_mk]
  /-
    case e_s.h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ι : Type u_4
    p : ι → P
    i₀ : ι
    v : V
    ⊢ Iff (Exists fun x => And (Exists fun x_1 => And (And (Membership.mem Set.uni …
  -/
  constructor
    /-
      case e_s.h.mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      ι : Type u_4
      p : ι → P
      i₀ : ι
      v : V
      ⊢ (Exists fun x => And (Exists fun x_1 => And (And (Membership.mem Set.univ x_ …
    -/
  · rintro ⟨x, ⟨i₁, ⟨⟨_, hi₁⟩, rfl⟩⟩, hv⟩
    /-
      case e_s.h.mp.intro.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      ι : Type u_4
      p : ι → P
      i₀ : ι
      v : V
      i₁ : ι
      left✝ : Membership.mem Set.univ i₁
      hi₁ : Not (Eq i₁ i₀)
      hv : Eq (VSub.vsub (p i₀) (p i₁)) v
      ⊢ Exists fun a => Exists fun h => Eq (VSub.vsub (p i₀) (p a)) v
    -/
    exact ⟨i₁, hi₁, hv⟩
    /-
      🎉 no goals
    -/
    /-
      case e_s.h.mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      ι : Type u_4
      p : ι → P
      i₀ : ι
      v : V
      ⊢ (Exists fun a => Exists fun h => Eq (VSub.vsub (p i₀) (p a)) v) → Exists fun …
    -/
  · exact fun ⟨i₁, hi₁, hv⟩ => ⟨p i₁, ⟨i₁, ⟨Set.mem_univ _, hi₁⟩, rfl⟩, hv⟩
    /-
      🎉 no goals
    -/


/-- The `vectorSpan` of an indexed family is the span of the pairwise subtractions with a given
point on the right, excluding the subtraction of that point from itself. -/
theorem vectorSpan_range_eq_span_range_vsub_right_ne (p : ι → P) (i₀ : ι) :
    vectorSpan k (Set.range p) =
      Submodule.span k (Set.range fun i : { x // x ≠ i₀ } => p i -ᵥ p i₀) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ι : Type u_4
    p : ι → P
    i₀ : ι
    ⊢ Eq (vectorSpan k (Set.range p)) (Submodule.span k (Set.range fun i => VSub.v …
  -/
  rw [← Set.image_univ, vectorSpan_image_eq_span_vsub_set_right_ne k _ (Set.mem_univ i₀)]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ι : Type u_4
    p : ι → P
    i₀ : ι
    ⊢ Eq (Submodule.span k (Set.image (fun x => VSub.vsub x (p i₀)) (Set.image p ( …
  -/
  congr with v
  simp only [Set.mem_range, Set.mem_image, Set.mem_diff, Set.mem_singleton_iff, Subtype.exists,
    Subtype.coe_mk]
  /-
    case e_s.h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    ι : Type u_4
    p : ι → P
    i₀ : ι
    v : V
    ⊢ Iff (Exists fun x => And (Exists fun x_1 => And (And (Membership.mem Set.uni …
  -/
  constructor
    /-
      case e_s.h.mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      ι : Type u_4
      p : ι → P
      i₀ : ι
      v : V
      ⊢ (Exists fun x => And (Exists fun x_1 => And (And (Membership.mem Set.univ x_ …
    -/
  · rintro ⟨x, ⟨i₁, ⟨⟨_, hi₁⟩, rfl⟩⟩, hv⟩
    /-
      case e_s.h.mp.intro.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      ι : Type u_4
      p : ι → P
      i₀ : ι
      v : V
      i₁ : ι
      left✝ : Membership.mem Set.univ i₁
      hi₁ : Not (Eq i₁ i₀)
      hv : Eq (VSub.vsub (p i₁) (p i₀)) v
      ⊢ Exists fun a => Exists fun h => Eq (VSub.vsub (p a) (p i₀)) v
    -/
    exact ⟨i₁, hi₁, hv⟩
    /-
      🎉 no goals
    -/
    /-
      case e_s.h.mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      ι : Type u_4
      p : ι → P
      i₀ : ι
      v : V
      ⊢ (Exists fun a => Exists fun h => Eq (VSub.vsub (p a) (p i₀)) v) → Exists fun …
    -/
  · exact fun ⟨i₁, hi₁, hv⟩ => ⟨p i₁, ⟨i₁, ⟨Set.mem_univ _, hi₁⟩, rfl⟩, hv⟩
    /-
      🎉 no goals
    -/


/-- The affine span of a set is nonempty if and only if that set is. -/
theorem affineSpan_nonempty : (affineSpan k s : Set P).Nonempty ↔ s.Nonempty :=
  spanPoints_nonempty k s


alias ⟨_, _root_.Set.Nonempty.affineSpan⟩ := affineSpan_nonempty


/-- The affine span of a nonempty set is nonempty. -/
instance [Nonempty s] : Nonempty (affineSpan k s) :=
  ((nonempty_coe_sort.1 ‹_›).affineSpan _).to_subtype


/-- The affine span of a set is `⊥` if and only if that set is empty. -/
@[simp]
theorem affineSpan_eq_bot : affineSpan k s = ⊥ ↔ s = ∅ := by
  rw [← not_iff_not, ← Ne, ← Ne, ← nonempty_iff_ne_bot, affineSpan_nonempty,
    nonempty_iff_ne_empty]


@[simp]
theorem bot_lt_affineSpan : ⊥ < affineSpan k s ↔ s.Nonempty := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    ⊢ Iff (LT.lt Bot.bot (affineSpan k s)) s.Nonempty
  -/
  rw [bot_lt_iff_ne_bot, nonempty_iff_ne_empty]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    ⊢ Iff (Ne (affineSpan k s) Bot.bot) (Ne s EmptyCollection.emptyCollection)
  -/
  exact (affineSpan_eq_bot _).not
  /-
    🎉 no goals
  -/


/-- An induction principle for span membership. If `p` holds for all elements of `s` and is
preserved under certain affine combinations, then `p` holds for all elements of the span of `s`. -/
theorem affineSpan_induction {x : P} {s : Set P} {p : P → Prop} (h : x ∈ affineSpan k s)
    (mem : ∀ x : P, x ∈ s → p x)
    (smul_vsub_vadd : ∀ (c : k) (u v w : P), p u → p v → p w → p (c • (u -ᵥ v) +ᵥ w)) : p x :=
  (affineSpan_le (Q := ⟨p, smul_vsub_vadd⟩)).mpr mem h


/-- A dependent version of `affineSpan_induction`. -/
@[elab_as_elim]
theorem affineSpan_induction' {s : Set P} {p : ∀ x, x ∈ affineSpan k s → Prop}
    (mem : ∀ (y) (hys : y ∈ s), p y (subset_affineSpan k _ hys))
    (smul_vsub_vadd :
      ∀ (c : k) (u hu v hv w hw),
        p u hu →
          p v hv → p w hw → p (c • (u -ᵥ v) +ᵥ w) (AffineSubspace.smul_vsub_vadd_mem _ _ hu hv hw))
    {x : P} (h : x ∈ affineSpan k s) : p x h := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p : (x : P) → Membership.mem (affineSpan k s) x → Prop
    mem : ∀ (y : P) (hys : Membership.mem s y), p y ⋯
    smul_vsub_vadd : ∀ (c : k) (u : P) (hu : Membership.mem (affineSpan k s) u) (v …
    x : P
    h : Membership.mem (affineSpan k s) x
    ⊢ p x h
  -/
  refine Exists.elim ?_ fun (hx : x ∈ affineSpan k s) (hc : p x hx) => hc
  -- Porting note: Lean couldn't infer the motive
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set P
    p : (x : P) → Membership.mem (affineSpan k s) x → Prop
    mem : ∀ (y : P) (hys : Membership.mem s y), p y ⋯
    smul_vsub_vadd : ∀ (c : k) (u : P) (hu : Membership.mem (affineSpan k s) u) (v …
    x : P
    h : Membership.mem (affineSpan k s) x
    ⊢ Exists fun x_1 => p x x_1
  -/
  refine affineSpan_induction (p := fun y => ∃ z, p y z) h ?_ ?_
    /-
      case refine_1
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : Set P
      p : (x : P) → Membership.mem (affineSpan k s) x → Prop
      mem : ∀ (y : P) (hys : Membership.mem s y), p y ⋯
      smul_vsub_vadd : ∀ (c : k) (u : P) (hu : Membership.mem (affineSpan k s) u) (v …
      x : P
      h : Membership.mem (affineSpan k s) x
      ⊢ ∀ (x : P), Membership.mem s x → (fun y => Exists fun z => p y z) x
    -/
  · exact fun y hy => ⟨subset_affineSpan _ _ hy, mem y hy⟩
    /-
      🎉 no goals
    -/
  · exact fun c u v w hu hv hw =>
      Exists.elim hu fun hu' hu =>
        Exists.elim hv fun hv' hv =>
          Exists.elim hw fun hw' hw =>
            ⟨AffineSubspace.smul_vsub_vadd_mem _ _ hu' hv' hw',
              smul_vsub_vadd _ _ _ _ _ _ _ hu hv hw⟩


/-- A set, considered as a subset of its spanned affine subspace, spans the whole subspace. -/
@[simp]
theorem affineSpan_coe_preimage_eq_top (A : Set P) [Nonempty A] :
    affineSpan k (((↑) : affineSpan k A → P) ⁻¹' A) = ⊤ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : Ring k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    A : Set P
    inst✝ : Nonempty ↑A
    ⊢ Eq (affineSpan k (Set.preimage Subtype.val A)) Top.top
  -/
  rw [eq_top_iff]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : Ring k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    A : Set P
    inst✝ : Nonempty ↑A
    ⊢ LE.le Top.top (affineSpan k (Set.preimage Subtype.val A))
  -/
  rintro ⟨x, hx⟩ -
  /-
    case mk
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝⁴ : Ring k
    inst✝³ : AddCommGroup V
    inst✝² : Module k V
    inst✝¹ : AddTorsor V P
    A : Set P
    inst✝ : Nonempty ↑A
    x : P
    hx : Membership.mem (affineSpan k A) x
    ⊢ Membership.mem ↑(affineSpan k (Set.preimage Subtype.val A)) ⟨x, hx⟩
  -/
  refine affineSpan_induction' (fun y hy ↦ ?_) (fun c u hu v hv w hw ↦ ?_) hx
    /-
      case mk.refine_1
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      A : Set P
      inst✝ : Nonempty ↑A
      x : P
      hx : Membership.mem (affineSpan k A) x
      y : P
      hy : Membership.mem A y
      ⊢ Membership.mem ↑(affineSpan k (Set.preimage Subtype.val A)) ⟨y, ⋯⟩
    -/
  · exact subset_affineSpan _ _ hy
    /-
      🎉 no goals
    -/
    /-
      case mk.refine_2
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝⁴ : Ring k
      inst✝³ : AddCommGroup V
      inst✝² : Module k V
      inst✝¹ : AddTorsor V P
      A : Set P
      inst✝ : Nonempty ↑A
      x : P
      hx : Membership.mem (affineSpan k A) x
      c : k
      u : P
      hu : Membership.mem (affineSpan k A) u
      v : P
      hv : Membership.mem (affineSpan k A) v
      w : P
      hw : Membership.mem (affineSpan k A) w
      ⊢ Membership.mem ↑(affineSpan k (Set.preimage Subtype.val A)) ⟨u, hu⟩ → Member …
    -/
  · exact AffineSubspace.smul_vsub_vadd_mem _ _
    /-
      🎉 no goals
    -/


/-- Suppose a set of vectors spans `V`.  Then a point `p`, together with those vectors added to `p`,
spans `P`. -/
theorem affineSpan_singleton_union_vadd_eq_top_of_span_eq_top {s : Set V} (p : P)
    (h : Submodule.span k (Set.range ((↑) : s → V)) = ⊤) :
    affineSpan k ({p} ∪ (fun v => v +ᵥ p) '' s) = ⊤ := by
  convert ext_of_direction_eq _
      ⟨p, mem_affineSpan k (Set.mem_union_left _ (Set.mem_singleton _)), mem_top k V p⟩
  rw [direction_affineSpan, direction_top,
    vectorSpan_eq_span_vsub_set_right k (Set.mem_union_left _ (Set.mem_singleton _) : p ∈ _),
    eq_top_iff, ← h]
  /-
    case convert_1
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set V
    p : P
    h : Eq (Submodule.span k (Set.range Subtype.val)) Top.top
    ⊢ LE.le (Submodule.span k (Set.range Subtype.val)) (Submodule.span k (Set.imag …
  -/
  apply Submodule.span_mono
  /-
    case convert_1.h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set V
    p : P
    h : Eq (Submodule.span k (Set.range Subtype.val)) Top.top
    ⊢ HasSubset.Subset (Set.range Subtype.val) (Set.image (fun x => VSub.vsub x p) …
  -/
  rintro v ⟨v', rfl⟩
  /-
    case convert_1.h.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set V
    p : P
    h : Eq (Submodule.span k (Set.range Subtype.val)) Top.top
    v' : Subtype fun x => Membership.mem s x
    ⊢ Membership.mem (Set.image (fun x => VSub.vsub x p) (Union.union (Singleton.s …
  -/
  use (v' : V) +ᵥ p
  /-
    case h
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : Set V
    p : P
    h : Eq (Submodule.span k (Set.range Subtype.val)) Top.top
    v' : Subtype fun x => Membership.mem s x
    ⊢ And (Membership.mem (Union.union (Singleton.singleton p) (Set.image (fun v = …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The `vectorSpan` of two points is the span of their difference. -/
theorem vectorSpan_pair (p₁ p₂ : P) : vectorSpan k ({p₁, p₂} : Set P) = k ∙ p₁ -ᵥ p₂ := by
  simp_rw [vectorSpan_eq_span_vsub_set_left k (mem_insert p₁ _), image_pair, vsub_self,
    Submodule.span_insert_zero]


/-- The `vectorSpan` of two points is the span of their difference (reversed). -/
theorem vectorSpan_pair_rev (p₁ p₂ : P) : vectorSpan k ({p₁, p₂} : Set P) = k ∙ p₂ -ᵥ p₁ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ : P
    ⊢ Eq (vectorSpan k (Insert.insert p₁ (Singleton.singleton p₂))) (Submodule.spa …
  -/
  rw [pair_comm, vectorSpan_pair]
  /-
    🎉 no goals
  -/


/-- The difference between two points lies in their `vectorSpan`. -/
theorem vsub_mem_vectorSpan_pair (p₁ p₂ : P) : p₁ -ᵥ p₂ ∈ vectorSpan k ({p₁, p₂} : Set P) :=
  vsub_mem_vectorSpan _ (Set.mem_insert _ _) (Set.mem_insert_of_mem _ (Set.mem_singleton _))


/-- The difference between two points (reversed) lies in their `vectorSpan`. -/
theorem vsub_rev_mem_vectorSpan_pair (p₁ p₂ : P) : p₂ -ᵥ p₁ ∈ vectorSpan k ({p₁, p₂} : Set P) :=
  vsub_mem_vectorSpan _ (Set.mem_insert_of_mem _ (Set.mem_singleton _)) (Set.mem_insert _ _)


/-- A multiple of the difference between two points lies in their `vectorSpan`. -/
theorem smul_vsub_mem_vectorSpan_pair (r : k) (p₁ p₂ : P) :
    r • (p₁ -ᵥ p₂) ∈ vectorSpan k ({p₁, p₂} : Set P) :=
  Submodule.smul_mem _ _ (vsub_mem_vectorSpan_pair k p₁ p₂)


/-- A multiple of the difference between two points (reversed) lies in their `vectorSpan`. -/
theorem smul_vsub_rev_mem_vectorSpan_pair (r : k) (p₁ p₂ : P) :
    r • (p₂ -ᵥ p₁) ∈ vectorSpan k ({p₁, p₂} : Set P) :=
  Submodule.smul_mem _ _ (vsub_rev_mem_vectorSpan_pair k p₁ p₂)


/-- A vector lies in the `vectorSpan` of two points if and only if it is a multiple of their
difference. -/
theorem mem_vectorSpan_pair {p₁ p₂ : P} {v : V} :
    v ∈ vectorSpan k ({p₁, p₂} : Set P) ↔ ∃ r : k, r • (p₁ -ᵥ p₂) = v := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ : P
    v : V
    ⊢ Iff (Membership.mem (vectorSpan k (Insert.insert p₁ (Singleton.singleton p₂) …
  -/
  rw [vectorSpan_pair, Submodule.mem_span_singleton]
  /-
    🎉 no goals
  -/


/-- A vector lies in the `vectorSpan` of two points if and only if it is a multiple of their
difference (reversed). -/
theorem mem_vectorSpan_pair_rev {p₁ p₂ : P} {v : V} :
    v ∈ vectorSpan k ({p₁, p₂} : Set P) ↔ ∃ r : k, r • (p₂ -ᵥ p₁) = v := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ : P
    v : V
    ⊢ Iff (Membership.mem (vectorSpan k (Insert.insert p₁ (Singleton.singleton p₂) …
  -/
  rw [vectorSpan_pair_rev, Submodule.mem_span_singleton]
  /-
    🎉 no goals
  -/


/-- The line between two points, as an affine subspace. -/
notation "line[" k ", " p₁ ", " p₂ "]" =>
  affineSpan k (insert p₁ (@singleton _ _ Set.instSingletonSet p₂))


/-- The first of two points lies in their affine span. -/
theorem left_mem_affineSpan_pair (p₁ p₂ : P) : p₁ ∈ line[k, p₁, p₂] :=
  mem_affineSpan _ (Set.mem_insert _ _)


/-- The second of two points lies in their affine span. -/
theorem right_mem_affineSpan_pair (p₁ p₂ : P) : p₂ ∈ line[k, p₁, p₂] :=
  mem_affineSpan _ (Set.mem_insert_of_mem _ (Set.mem_singleton _))


/-- A combination of two points expressed with `lineMap` lies in their affine span. -/
theorem AffineMap.lineMap_mem_affineSpan_pair (r : k) (p₁ p₂ : P) :
    AffineMap.lineMap p₁ p₂ r ∈ line[k, p₁, p₂] :=
  AffineMap.lineMap_mem _ (left_mem_affineSpan_pair _ _ _) (right_mem_affineSpan_pair _ _ _)


/-- A combination of two points expressed with `lineMap` (with the two points reversed) lies in
their affine span. -/
theorem AffineMap.lineMap_rev_mem_affineSpan_pair (r : k) (p₁ p₂ : P) :
    AffineMap.lineMap p₂ p₁ r ∈ line[k, p₁, p₂] :=
  AffineMap.lineMap_mem _ (right_mem_affineSpan_pair _ _ _) (left_mem_affineSpan_pair _ _ _)


/-- A multiple of the difference of two points added to the first point lies in their affine
span. -/
theorem smul_vsub_vadd_mem_affineSpan_pair (r : k) (p₁ p₂ : P) :
    r • (p₂ -ᵥ p₁) +ᵥ p₁ ∈ line[k, p₁, p₂] :=
  AffineMap.lineMap_mem_affineSpan_pair _ _ _


/-- A multiple of the difference of two points added to the second point lies in their affine
span. -/
theorem smul_vsub_rev_vadd_mem_affineSpan_pair (r : k) (p₁ p₂ : P) :
    r • (p₁ -ᵥ p₂) +ᵥ p₂ ∈ line[k, p₁, p₂] :=
  AffineMap.lineMap_rev_mem_affineSpan_pair _ _ _


/-- A vector added to the first point lies in the affine span of two points if and only if it is
a multiple of their difference. -/
theorem vadd_left_mem_affineSpan_pair {p₁ p₂ : P} {v : V} :
    v +ᵥ p₁ ∈ line[k, p₁, p₂] ↔ ∃ r : k, r • (p₂ -ᵥ p₁) = v := by
  rw [vadd_mem_iff_mem_direction _ (left_mem_affineSpan_pair _ _ _), direction_affineSpan,
    mem_vectorSpan_pair_rev]


/-- A vector added to the second point lies in the affine span of two points if and only if it is
a multiple of their difference. -/
theorem vadd_right_mem_affineSpan_pair {p₁ p₂ : P} {v : V} :
    v +ᵥ p₂ ∈ line[k, p₁, p₂] ↔ ∃ r : k, r • (p₁ -ᵥ p₂) = v := by
  rw [vadd_mem_iff_mem_direction _ (right_mem_affineSpan_pair _ _ _), direction_affineSpan,
    mem_vectorSpan_pair]


/-- The span of two points that lie in an affine subspace is contained in that subspace. -/
theorem affineSpan_pair_le_of_mem_of_mem {p₁ p₂ : P} {s : AffineSubspace k P} (hp₁ : p₁ ∈ s)
    (hp₂ : p₂ ∈ s) : line[k, p₁, p₂] ≤ s := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ : P
    s : AffineSubspace k P
    hp₁ : Membership.mem s p₁
    hp₂ : Membership.mem s p₂
    ⊢ LE.le (affineSpan k (Insert.insert p₁ (Singleton.singleton p₂))) s
  -/
  rw [affineSpan_le, Set.insert_subset_iff, Set.singleton_subset_iff]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p₁ p₂ : P
    s : AffineSubspace k P
    hp₁ : Membership.mem s p₁
    hp₂ : Membership.mem s p₂
    ⊢ And (Membership.mem (↑s) p₁) (Membership.mem (↑s) p₂)
  -/
  exact ⟨hp₁, hp₂⟩
  /-
    🎉 no goals
  -/


/-- One line is contained in another differing in the first point if the first point of the first
line is contained in the second line. -/
theorem affineSpan_pair_le_of_left_mem {p₁ p₂ p₃ : P} (h : p₁ ∈ line[k, p₂, p₃]) :
    line[k, p₁, p₃] ≤ line[k, p₂, p₃] :=
  affineSpan_pair_le_of_mem_of_mem h (right_mem_affineSpan_pair _ _ _)


/-- One line is contained in another differing in the second point if the second point of the
first line is contained in the second line. -/
theorem affineSpan_pair_le_of_right_mem {p₁ p₂ p₃ : P} (h : p₁ ∈ line[k, p₂, p₃]) :
    line[k, p₂, p₁] ≤ line[k, p₂, p₃] :=
  affineSpan_pair_le_of_mem_of_mem (left_mem_affineSpan_pair _ _ _) h


/-- `affineSpan` is monotone. -/
@[gcongr, mono]
theorem affineSpan_mono {s₁ s₂ : Set P} (h : s₁ ⊆ s₂) : affineSpan k s₁ ≤ affineSpan k s₂ :=
  spanPoints_subset_coe_of_subset_coe (Set.Subset.trans h (subset_affineSpan k _))


/-- Taking the affine span of a set, adding a point and taking the span again produces the same
results as adding the point to the set and taking the span. -/
theorem affineSpan_insert_affineSpan (p : P) (ps : Set P) :
    affineSpan k (insert p (affineSpan k ps : Set P)) = affineSpan k (insert p ps) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p : P
    ps : Set P
    ⊢ Eq (affineSpan k (Insert.insert p ↑(affineSpan k ps))) (affineSpan k (Insert …
  -/
  rw [Set.insert_eq, Set.insert_eq, span_union, span_union, affineSpan_coe]
  /-
    🎉 no goals
  -/


/-- If a point is in the affine span of a set, adding it to that set does not change the affine
span. -/
theorem affineSpan_insert_eq_affineSpan {p : P} {ps : Set P} (h : p ∈ affineSpan k ps) :
    affineSpan k (insert p ps) = affineSpan k ps := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p : P
    ps : Set P
    h : Membership.mem (affineSpan k ps) p
    ⊢ Eq (affineSpan k (Insert.insert p ps)) (affineSpan k ps)
  -/
  rw [← mem_coe] at h
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p : P
    ps : Set P
    h : Membership.mem (↑(affineSpan k ps)) p
    ⊢ Eq (affineSpan k (Insert.insert p ps)) (affineSpan k ps)
  -/
  rw [← affineSpan_insert_affineSpan, Set.insert_eq_of_mem h, affineSpan_coe]
  /-
    🎉 no goals
  -/


/-- If a point is in the affine span of a set, adding it to that set does not change the vector
span. -/
theorem vectorSpan_insert_eq_vectorSpan {p : P} {ps : Set P} (h : p ∈ affineSpan k ps) :
    vectorSpan k (insert p ps) = vectorSpan k ps := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    p : P
    ps : Set P
    h : Membership.mem (affineSpan k ps) p
    ⊢ Eq (vectorSpan k (Insert.insert p ps)) (vectorSpan k ps)
  -/
  simp_rw [← direction_affineSpan, affineSpan_insert_eq_affineSpan _ h]
  /-
    🎉 no goals
  -/


/-- When the affine space is also a vector space, the affine span is contained within the linear
span. -/
lemma affineSpan_le_toAffineSubspace_span {s : Set V} :
    affineSpan k s ≤ (Submodule.span k s).toAffineSubspace := by
  /-
    k : Type u_1
    V : Type u_2
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s : Set V
    ⊢ LE.le (affineSpan k s) (Submodule.span k s).toAffineSubspace
  -/
  intro x hx
  /-
    k : Type u_1
    V : Type u_2
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s : Set V
    x : V
    hx : Membership.mem (↑(affineSpan k s)) x
    ⊢ Membership.mem (↑(Submodule.span k s).toAffineSubspace) x
  -/
  simp only [SetLike.mem_coe, Submodule.mem_toAffineSubspace]
  induction hx using affineSpan_induction' with
  | mem x hx => exact Submodule.subset_span hx
  | smul_vsub_vadd c u _ v _ w _ hu hv hw =>
    simp only [vsub_eq_sub, vadd_eq_add]
    apply Submodule.add_mem _ _ hw
    exact Submodule.smul_mem _ _ (Submodule.sub_mem _ hu hv)


lemma affineSpan_subset_span {s : Set V} :
    (affineSpan k s : Set V) ⊆  Submodule.span k s :=
  affineSpan_le_toAffineSubspace_span

-- TODO: We want this to be simp, but `affineSpan` gets simped away to `spanPoints`!
-- Let's delete `spanPoints`

lemma affineSpan_insert_zero (s : Set V) :
    (affineSpan k (insert 0 s) : Set V) = Submodule.span k s := by
  /-
    k : Type u_1
    V : Type u_2
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s : Set V
    ⊢ Eq ↑(affineSpan k (Insert.insert 0 s)) ↑(Submodule.span k s)
  -/
  rw [← Submodule.span_insert_zero]
  /-
    k : Type u_1
    V : Type u_2
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s : Set V
    ⊢ Eq ↑(affineSpan k (Insert.insert 0 s)) ↑(Submodule.span k (Insert.insert 0 s))
  -/
  refine affineSpan_subset_span.antisymm ?_
  /-
    k : Type u_1
    V : Type u_2
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s : Set V
    ⊢ HasSubset.Subset ↑(Submodule.span k (Insert.insert 0 s)) ↑(affineSpan k (Ins …
  -/
  rw [← vectorSpan_add_self, vectorSpan_def]
  /-
    k : Type u_1
    V : Type u_2
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s : Set V
    ⊢ HasSubset.Subset (↑(Submodule.span k (Insert.insert 0 s))) (HAdd.hAdd (↑(Sub …
  -/
  refine Subset.trans ?_ <| subset_add_left _ <| mem_insert ..
  /-
    k : Type u_1
    V : Type u_2
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s : Set V
    ⊢ HasSubset.Subset ↑(Submodule.span k (Insert.insert 0 s)) ↑(Submodule.span k  …
  -/
  gcongr
  /-
    case a.h
    k : Type u_1
    V : Type u_2
    inst✝² : Ring k
    inst✝¹ : AddCommGroup V
    inst✝ : Module k V
    s : Set V
    ⊢ HasSubset.Subset (Insert.insert 0 s) (VSub.vsub (Insert.insert 0 s) (Insert. …
  -/
  exact subset_sub_left <| mem_insert ..
  /-
    🎉 no goals
  -/


/-- The direction of the sup of two nonempty affine subspaces is the sup of the two directions and
of any one difference between points in the two subspaces. -/
theorem direction_sup {s1 s2 : AffineSubspace k P} {p1 p2 : P} (hp1 : p1 ∈ s1) (hp2 : p2 ∈ s2) :
    (s1 ⊔ s2).direction = s1.direction ⊔ s2.direction ⊔ k ∙ p2 -ᵥ p1 := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s1 s2 : AffineSubspace k P
    p1 p2 : P
    hp1 : Membership.mem s1 p1
    hp2 : Membership.mem s2 p2
    ⊢ Eq (Max.max s1 s2).direction (Max.max (Max.max s1.direction s2.direction) (S …
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s1 s2 : AffineSubspace k P
      p1 p2 : P
      hp1 : Membership.mem s1 p1
      hp2 : Membership.mem s2 p2
      ⊢ LE.le (Max.max s1 s2).direction (Max.max (Max.max s1.direction s2.direction) …
    -/
  · change (affineSpan k ((s1 : Set P) ∪ s2)).direction ≤ _
    /-
      case refine_1
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s1 s2 : AffineSubspace k P
      p1 p2 : P
      hp1 : Membership.mem s1 p1
      hp2 : Membership.mem s2 p2
      ⊢ LE.le (affineSpan k (Union.union ↑s1 ↑s2)).direction (Max.max (Max.max s1.di …
    -/
    rw [← mem_coe] at hp1
    rw [direction_affineSpan, vectorSpan_eq_span_vsub_set_right k (Set.mem_union_left _ hp1),
      Submodule.span_le]
    /-
      case refine_1
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s1 s2 : AffineSubspace k P
      p1 p2 : P
      hp1 : Membership.mem (↑s1) p1
      hp2 : Membership.mem s2 p2
      ⊢ HasSubset.Subset (Set.image (fun x => VSub.vsub x p1) (Union.union ↑s1 ↑s2)) …
    -/
    rintro v ⟨p3, hp3, rfl⟩
    /-
      case refine_1.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s1 s2 : AffineSubspace k P
      p1 p2 : P
      hp1 : Membership.mem (↑s1) p1
      hp2 : Membership.mem s2 p2
      p3 : P
      hp3 : Membership.mem (Union.union ↑s1 ↑s2) p3
      ⊢ Membership.mem (↑(Max.max (Max.max s1.direction s2.direction) (Submodule.spa …
    -/
    cases' hp3 with hp3 hp3
      /-
        case refine_1.intro.intro.inl
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : Ring k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s1 s2 : AffineSubspace k P
        p1 p2 : P
        hp1 : Membership.mem (↑s1) p1
        hp2 : Membership.mem s2 p2
        p3 : P
        hp3 : Membership.mem (↑s1) p3
        ⊢ Membership.mem (↑(Max.max (Max.max s1.direction s2.direction) (Submodule.spa …
      -/
    · rw [sup_assoc, sup_comm, SetLike.mem_coe, Submodule.mem_sup]
      /-
        case refine_1.intro.intro.inl
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : Ring k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s1 s2 : AffineSubspace k P
        p1 p2 : P
        hp1 : Membership.mem (↑s1) p1
        hp2 : Membership.mem s2 p2
        p3 : P
        hp3 : Membership.mem (↑s1) p3
        ⊢ Exists fun y => And (Membership.mem (Max.max s2.direction (Submodule.span k  …
      -/
      use 0, Submodule.zero_mem _, p3 -ᵥ p1, vsub_mem_direction hp3 hp1
      /-
        case right
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : Ring k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s1 s2 : AffineSubspace k P
        p1 p2 : P
        hp1 : Membership.mem (↑s1) p1
        hp2 : Membership.mem s2 p2
        p3 : P
        hp3 : Membership.mem (↑s1) p3
        ⊢ Eq (HAdd.hAdd 0 (VSub.vsub p3 p1)) ((fun x => VSub.vsub x p1) p3)
      -/
      rw [zero_add]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.inr
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : Ring k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s1 s2 : AffineSubspace k P
        p1 p2 : P
        hp1 : Membership.mem (↑s1) p1
        hp2 : Membership.mem s2 p2
        p3 : P
        hp3 : Membership.mem (↑s2) p3
        ⊢ Membership.mem (↑(Max.max (Max.max s1.direction s2.direction) (Submodule.spa …
      -/
    · rw [sup_assoc, SetLike.mem_coe, Submodule.mem_sup]
      /-
        case refine_1.intro.intro.inr
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : Ring k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s1 s2 : AffineSubspace k P
        p1 p2 : P
        hp1 : Membership.mem (↑s1) p1
        hp2 : Membership.mem s2 p2
        p3 : P
        hp3 : Membership.mem (↑s2) p3
        ⊢ Exists fun y => And (Membership.mem s1.direction y) (Exists fun z => And (Me …
      -/
      use 0, Submodule.zero_mem _, p3 -ᵥ p1
      /-
        case h
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : Ring k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s1 s2 : AffineSubspace k P
        p1 p2 : P
        hp1 : Membership.mem (↑s1) p1
        hp2 : Membership.mem s2 p2
        p3 : P
        hp3 : Membership.mem (↑s2) p3
        ⊢ And (Membership.mem (Max.max s2.direction (Submodule.span k (Singleton.singl …
      -/
      rw [and_comm, zero_add]
      /-
        case h
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : Ring k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s1 s2 : AffineSubspace k P
        p1 p2 : P
        hp1 : Membership.mem (↑s1) p1
        hp2 : Membership.mem s2 p2
        p3 : P
        hp3 : Membership.mem (↑s2) p3
        ⊢ And (Eq (VSub.vsub p3 p1) ((fun x => VSub.vsub x p1) p3)) (Membership.mem (M …
      -/
      use rfl
      /-
        case right
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : Ring k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s1 s2 : AffineSubspace k P
        p1 p2 : P
        hp1 : Membership.mem (↑s1) p1
        hp2 : Membership.mem s2 p2
        p3 : P
        hp3 : Membership.mem (↑s2) p3
        ⊢ Membership.mem (Max.max s2.direction (Submodule.span k (Singleton.singleton  …
      -/
      rw [← vsub_add_vsub_cancel p3 p2 p1, Submodule.mem_sup]
      /-
        case right
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : Ring k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s1 s2 : AffineSubspace k P
        p1 p2 : P
        hp1 : Membership.mem (↑s1) p1
        hp2 : Membership.mem s2 p2
        p3 : P
        hp3 : Membership.mem (↑s2) p3
        ⊢ Exists fun y => And (Membership.mem s2.direction y) (Exists fun z => And (Me …
      -/
      use p3 -ᵥ p2, vsub_mem_direction hp3 hp2, p2 -ᵥ p1, Submodule.mem_span_singleton_self _
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s1 s2 : AffineSubspace k P
      p1 p2 : P
      hp1 : Membership.mem s1 p1
      hp2 : Membership.mem s2 p2
      ⊢ LE.le (Max.max (Max.max s1.direction s2.direction) (Submodule.span k (Single …
    -/
  · refine sup_le (sup_direction_le _ _) ?_
    /-
      case refine_2
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s1 s2 : AffineSubspace k P
      p1 p2 : P
      hp1 : Membership.mem s1 p1
      hp2 : Membership.mem s2 p2
      ⊢ LE.le (Submodule.span k (Singleton.singleton (VSub.vsub p2 p1))) (Max.max s1 …
    -/
    rw [direction_eq_vectorSpan, vectorSpan_def]
    exact
      sInf_le_sInf fun p hp =>
        Set.Subset.trans
          (Set.singleton_subset_iff.2
            (vsub_mem_vsub (mem_spanPoints k p2 _ (Set.mem_union_right _ hp2))
              (mem_spanPoints k p1 _ (Set.mem_union_left _ hp1))))
          hp


/-- The direction of the span of the result of adding a point to a nonempty affine subspace is the
sup of the direction of that subspace and of any one difference between that point and a point in
the subspace. -/
theorem direction_affineSpan_insert {s : AffineSubspace k P} {p1 p2 : P} (hp1 : p1 ∈ s) :
    (affineSpan k (insert p2 (s : Set P))).direction =
    Submodule.span k {p2 -ᵥ p1} ⊔ s.direction := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p1 p2 : P
    hp1 : Membership.mem s p1
    ⊢ Eq (affineSpan k (Insert.insert p2 ↑s)).direction (Max.max (Submodule.span k …
  -/
  rw [sup_comm, ← Set.union_singleton, ← coe_affineSpan_singleton k V p2]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p1 p2 : P
    hp1 : Membership.mem s p1
    ⊢ Eq (affineSpan k (Union.union ↑s ↑(affineSpan k (Singleton.singleton p2)))). …
  -/
  change (s ⊔ affineSpan k {p2}).direction = _
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p1 p2 : P
    hp1 : Membership.mem s p1
    ⊢ Eq (Max.max s (affineSpan k (Singleton.singleton p2))).direction (Max.max s. …
  -/
  rw [direction_sup hp1 (mem_affineSpan k (Set.mem_singleton _)), direction_affineSpan]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p1 p2 : P
    hp1 : Membership.mem s p1
    ⊢ Eq (Max.max (Max.max s.direction (vectorSpan k (Singleton.singleton p2))) (S …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given a point `p1` in an affine subspace `s`, and a point `p2`, a point `p` is in the span of
`s` with `p2` added if and only if it is a multiple of `p2 -ᵥ p1` added to a point in `s`. -/
theorem mem_affineSpan_insert_iff {s : AffineSubspace k P} {p1 : P} (hp1 : p1 ∈ s) (p2 p : P) :
    p ∈ affineSpan k (insert p2 (s : Set P)) ↔
      ∃ r : k, ∃ p0 ∈ s, p = r • (p2 -ᵥ p1 : V) +ᵥ p0 := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p1 : P
    hp1 : Membership.mem s p1
    p2 p : P
    ⊢ Iff (Membership.mem (affineSpan k (Insert.insert p2 ↑s)) p) (Exists fun r => …
  -/
  rw [← mem_coe] at hp1
  rw [← vsub_right_mem_direction_iff_mem (mem_affineSpan k (Set.mem_insert_of_mem _ hp1)),
    direction_affineSpan_insert hp1, Submodule.mem_sup]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    p1 : P
    hp1 : Membership.mem (↑s) p1
    p2 p : P
    ⊢ Iff (Exists fun y => And (Membership.mem (Submodule.span k (Singleton.single …
  -/
  constructor
    /-
      case mp
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p1 : P
      hp1 : Membership.mem (↑s) p1
      p2 p : P
      ⊢ (Exists fun y => And (Membership.mem (Submodule.span k (Singleton.singleton  …
    -/
  · rintro ⟨v1, hv1, v2, hv2, hp⟩
    /-
      case mp.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p1 : P
      hp1 : Membership.mem (↑s) p1
      p2 p : P
      v1 : V
      hv1 : Membership.mem (Submodule.span k (Singleton.singleton (VSub.vsub p2 p1)) …
      v2 : V
      hv2 : Membership.mem s.direction v2
      hp : Eq (HAdd.hAdd v1 v2) (VSub.vsub p p1)
      ⊢ Exists fun r => Exists fun p0 => And (Membership.mem s p0) (Eq p (HVAdd.hVAd …
    -/
    rw [Submodule.mem_span_singleton] at hv1
    /-
      case mp.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p1 : P
      hp1 : Membership.mem (↑s) p1
      p2 p : P
      v1 : V
      hv1 : Exists fun a => Eq (HSMul.hSMul a (VSub.vsub p2 p1)) v1
      v2 : V
      hv2 : Membership.mem s.direction v2
      hp : Eq (HAdd.hAdd v1 v2) (VSub.vsub p p1)
      ⊢ Exists fun r => Exists fun p0 => And (Membership.mem s p0) (Eq p (HVAdd.hVAd …
    -/
    rcases hv1 with ⟨r, rfl⟩
    /-
      case mp.intro.intro.intro.intro.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p1 : P
      hp1 : Membership.mem (↑s) p1
      p2 p : P
      v2 : V
      hv2 : Membership.mem s.direction v2
      r : k
      hp : Eq (HAdd.hAdd (HSMul.hSMul r (VSub.vsub p2 p1)) v2) (VSub.vsub p p1)
      ⊢ Exists fun r => Exists fun p0 => And (Membership.mem s p0) (Eq p (HVAdd.hVAd …
    -/
    use r, v2 +ᵥ p1, vadd_mem_of_mem_direction hv2 hp1
    /-
      case right
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p1 : P
      hp1 : Membership.mem (↑s) p1
      p2 p : P
      v2 : V
      hv2 : Membership.mem s.direction v2
      r : k
      hp : Eq (HAdd.hAdd (HSMul.hSMul r (VSub.vsub p2 p1)) v2) (VSub.vsub p p1)
      ⊢ Eq p (HVAdd.hVAdd (HSMul.hSMul r (VSub.vsub p2 p1)) (HVAdd.hVAdd v2 p1))
    -/
    symm at hp
    /-
      case right
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p1 : P
      hp1 : Membership.mem (↑s) p1
      p2 p : P
      v2 : V
      hv2 : Membership.mem s.direction v2
      r : k
      hp : Eq (VSub.vsub p p1) (HAdd.hAdd (HSMul.hSMul r (VSub.vsub p2 p1)) v2)
      ⊢ Eq p (HVAdd.hVAdd (HSMul.hSMul r (VSub.vsub p2 p1)) (HVAdd.hVAdd v2 p1))
    -/
    rw [← sub_eq_zero, ← vsub_vadd_eq_vsub_sub, vsub_eq_zero_iff_eq] at hp
    /-
      case right
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p1 : P
      hp1 : Membership.mem (↑s) p1
      p2 p : P
      v2 : V
      hv2 : Membership.mem s.direction v2
      r : k
      hp : Eq p (HVAdd.hVAdd (HAdd.hAdd (HSMul.hSMul r (VSub.vsub p2 p1)) v2) p1)
      ⊢ Eq p (HVAdd.hVAdd (HSMul.hSMul r (VSub.vsub p2 p1)) (HVAdd.hVAdd v2 p1))
    -/
    rw [hp, vadd_vadd]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p1 : P
      hp1 : Membership.mem (↑s) p1
      p2 p : P
      ⊢ (Exists fun r => Exists fun p0 => And (Membership.mem s p0) (Eq p (HVAdd.hVA …
    -/
  · rintro ⟨r, p3, hp3, rfl⟩
    use r • (p2 -ᵥ p1), Submodule.mem_span_singleton.2 ⟨r, rfl⟩, p3 -ᵥ p1,
      vsub_mem_direction hp3 hp1
    /-
      case right
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s : AffineSubspace k P
      p1 : P
      hp1 : Membership.mem (↑s) p1
      p2 : P
      r : k
      p3 : P
      hp3 : Membership.mem s p3
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul r (VSub.vsub p2 p1)) (VSub.vsub p3 p1)) (VSub.vsu …
    -/
    rw [vadd_vsub_assoc]
    /-
      🎉 no goals
    -/


@[simp]
theorem AffineMap.vectorSpan_image_eq_submodule_map {s : Set P₁} :
    Submodule.map f.linear (vectorSpan k s) = vectorSpan k (f '' s) := by
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    f : AffineMap k P₁ P₂
    s : Set P₁
    ⊢ Eq (Submodule.map f.linear (vectorSpan k s)) (vectorSpan k (Set.image (⇑f) s))
  -/
  rw [vectorSpan_def, vectorSpan_def, f.image_vsub_image, Submodule.span_image]
  /-
    🎉 no goals
  -/
  -- Porting note: Lean unfolds things too far with `simp` here.


/-- The image of an affine subspace under an affine map as an affine subspace. -/
def map (s : AffineSubspace k P₁) : AffineSubspace k P₂ where
  carrier := f '' s
  smul_vsub_vadd_mem := by
    /-
      k : Type u_1
      V₁ : Type u_2
      P₁ : Type u_3
      V₂ : Type u_4
      P₂ : Type u_5
      V₃ : Type u_6
      P₃ : Type u_7
      inst✝⁹ : Ring k
      inst✝⁸ : AddCommGroup V₁
      inst✝⁷ : Module k V₁
      inst✝⁶ : AddTorsor V₁ P₁
      inst✝⁵ : AddCommGroup V₂
      inst✝⁴ : Module k V₂
      inst✝³ : AddTorsor V₂ P₂
      inst✝² : AddCommGroup V₃
      inst✝¹ : Module k V₃
      inst✝ : AddTorsor V₃ P₃
      f : AffineMap k P₁ P₂
      s : AffineSubspace k P₁
      ⊢ ∀ (c : k) {p1 p2 p3 : P₂}, Membership.mem (Set.image ⇑f ↑s) p1 → Membership. …
    -/
    rintro t - - - ⟨p₁, h₁, rfl⟩ ⟨p₂, h₂, rfl⟩ ⟨p₃, h₃, rfl⟩
    /-
      case intro.intro.intro.intro.intro.intro
      k : Type u_1
      V₁ : Type u_2
      P₁ : Type u_3
      V₂ : Type u_4
      P₂ : Type u_5
      V₃ : Type u_6
      P₃ : Type u_7
      inst✝⁹ : Ring k
      inst✝⁸ : AddCommGroup V₁
      inst✝⁷ : Module k V₁
      inst✝⁶ : AddTorsor V₁ P₁
      inst✝⁵ : AddCommGroup V₂
      inst✝⁴ : Module k V₂
      inst✝³ : AddTorsor V₂ P₂
      inst✝² : AddCommGroup V₃
      inst✝¹ : Module k V₃
      inst✝ : AddTorsor V₃ P₃
      f : AffineMap k P₁ P₂
      s : AffineSubspace k P₁
      t : k
      p₁ : P₁
      h₁ : Membership.mem (↑s) p₁
      p₂ : P₁
      h₂ : Membership.mem (↑s) p₂
      p₃ : P₁
      h₃ : Membership.mem (↑s) p₃
      ⊢ Membership.mem (Set.image ⇑f ↑s) (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub (f p …
    -/
    use t • (p₁ -ᵥ p₂) +ᵥ p₃
    suffices t • (p₁ -ᵥ p₂) +ᵥ p₃ ∈ s by
    { simp only [SetLike.mem_coe, true_and, this]
      rw [AffineMap.map_vadd, map_smul, AffineMap.linearMap_vsub] }
    /-
      case h
      k : Type u_1
      V₁ : Type u_2
      P₁ : Type u_3
      V₂ : Type u_4
      P₂ : Type u_5
      V₃ : Type u_6
      P₃ : Type u_7
      inst✝⁹ : Ring k
      inst✝⁸ : AddCommGroup V₁
      inst✝⁷ : Module k V₁
      inst✝⁶ : AddTorsor V₁ P₁
      inst✝⁵ : AddCommGroup V₂
      inst✝⁴ : Module k V₂
      inst✝³ : AddTorsor V₂ P₂
      inst✝² : AddCommGroup V₃
      inst✝¹ : Module k V₃
      inst✝ : AddTorsor V₃ P₃
      f : AffineMap k P₁ P₂
      s : AffineSubspace k P₁
      t : k
      p₁ : P₁
      h₁ : Membership.mem (↑s) p₁
      p₂ : P₁
      h₂ : Membership.mem (↑s) p₂
      p₃ : P₁
      h₃ : Membership.mem (↑s) p₃
      ⊢ Membership.mem s (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub p₁ p₂)) p₃)
    -/
    exact s.smul_vsub_vadd_mem t h₁ h₂ h₃
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_map (s : AffineSubspace k P₁) : (s.map f : Set P₂) = f '' s :=
  rfl


@[simp]
theorem mem_map {f : P₁ →ᵃ[k] P₂} {x : P₂} {s : AffineSubspace k P₁} :
    x ∈ s.map f ↔ ∃ y ∈ s, f y = x :=
  Iff.rfl


theorem mem_map_of_mem {x : P₁} {s : AffineSubspace k P₁} (h : x ∈ s) : f x ∈ s.map f :=
  Set.mem_image_of_mem _ h

-- The simpNF linter says that the LHS can be simplified via `AffineSubspace.mem_map`.
-- However this is a higher priority lemma.
-- https://github.com/leanprover/std4/issues/207

@[simp 1100, nolint simpNF]
theorem mem_map_iff_mem_of_injective {f : P₁ →ᵃ[k] P₂} {x : P₁} {s : AffineSubspace k P₁}
    (hf : Function.Injective f) : f x ∈ s.map f ↔ x ∈ s :=
  hf.mem_set_image


@[simp]
theorem map_bot : (⊥ : AffineSubspace k P₁).map f = ⊥ :=
  coe_injective <| image_empty f


@[simp]
theorem map_eq_bot_iff {s : AffineSubspace k P₁} : s.map f = ⊥ ↔ s = ⊥ := by
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    f : AffineMap k P₁ P₂
    s : AffineSubspace k P₁
    ⊢ Iff (Eq (AffineSubspace.map f s) Bot.bot) (Eq s Bot.bot)
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      k : Type u_1
      V₁ : Type u_2
      P₁ : Type u_3
      V₂ : Type u_4
      P₂ : Type u_5
      inst✝⁶ : Ring k
      inst✝⁵ : AddCommGroup V₁
      inst✝⁴ : Module k V₁
      inst✝³ : AddTorsor V₁ P₁
      inst✝² : AddCommGroup V₂
      inst✝¹ : Module k V₂
      inst✝ : AddTorsor V₂ P₂
      f : AffineMap k P₁ P₂
      s : AffineSubspace k P₁
      h : Eq (AffineSubspace.map f s) Bot.bot
      ⊢ Eq s Bot.bot
    -/
  · rwa [← coe_eq_bot_iff, coe_map, image_eq_empty, coe_eq_bot_iff] at h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      k : Type u_1
      V₁ : Type u_2
      P₁ : Type u_3
      V₂ : Type u_4
      P₂ : Type u_5
      inst✝⁶ : Ring k
      inst✝⁵ : AddCommGroup V₁
      inst✝⁴ : Module k V₁
      inst✝³ : AddTorsor V₁ P₁
      inst✝² : AddCommGroup V₂
      inst✝¹ : Module k V₂
      inst✝ : AddTorsor V₂ P₂
      f : AffineMap k P₁ P₂
      s : AffineSubspace k P₁
      h : Eq s Bot.bot
      ⊢ Eq (AffineSubspace.map f s) Bot.bot
    -/
  · rw [h, map_bot]
    /-
      🎉 no goals
    -/


@[simp]
theorem map_id (s : AffineSubspace k P₁) : s.map (AffineMap.id k P₁) = s :=
  coe_injective <| image_id _


theorem map_map (s : AffineSubspace k P₁) (f : P₁ →ᵃ[k] P₂) (g : P₂ →ᵃ[k] P₃) :
    (s.map f).map g = s.map (g.comp f) :=
  coe_injective <| image_image _ _ _


@[simp]
theorem map_direction (s : AffineSubspace k P₁) :
    (s.map f).direction = s.direction.map f.linear := by
  rw [direction_eq_vectorSpan, direction_eq_vectorSpan, coe_map,
    AffineMap.vectorSpan_image_eq_submodule_map]
  -- Porting note: again, Lean unfolds too aggressively with `simp`


theorem map_span (s : Set P₁) : (affineSpan k s).map f = affineSpan k (f '' s) := by
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    f : AffineMap k P₁ P₂
    s : Set P₁
    ⊢ Eq (AffineSubspace.map f (affineSpan k s)) (affineSpan k (Set.image (⇑f) s))
  -/
  rcases s.eq_empty_or_nonempty with (rfl | ⟨p, hp⟩)
    /-
      case inl
      k : Type u_1
      V₁ : Type u_2
      P₁ : Type u_3
      V₂ : Type u_4
      P₂ : Type u_5
      inst✝⁶ : Ring k
      inst✝⁵ : AddCommGroup V₁
      inst✝⁴ : Module k V₁
      inst✝³ : AddTorsor V₁ P₁
      inst✝² : AddCommGroup V₂
      inst✝¹ : Module k V₂
      inst✝ : AddTorsor V₂ P₂
      f : AffineMap k P₁ P₂
      ⊢ Eq (AffineSubspace.map f (affineSpan k EmptyCollection.emptyCollection)) (af …
    -/
  · rw [image_empty, span_empty, span_empty, map_bot]
    /-
      🎉 no goals
    -/
    -- Porting note: I don't know exactly why this `simp` was broken.
  /-
    case inr.intro
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    f : AffineMap k P₁ P₂
    s : Set P₁
    p : P₁
    hp : Membership.mem s p
    ⊢ Eq (AffineSubspace.map f (affineSpan k s)) (affineSpan k (Set.image (⇑f) s))
  -/
  apply ext_of_direction_eq
    /-
      case inr.intro.hd
      k : Type u_1
      V₁ : Type u_2
      P₁ : Type u_3
      V₂ : Type u_4
      P₂ : Type u_5
      inst✝⁶ : Ring k
      inst✝⁵ : AddCommGroup V₁
      inst✝⁴ : Module k V₁
      inst✝³ : AddTorsor V₁ P₁
      inst✝² : AddCommGroup V₂
      inst✝¹ : Module k V₂
      inst✝ : AddTorsor V₂ P₂
      f : AffineMap k P₁ P₂
      s : Set P₁
      p : P₁
      hp : Membership.mem s p
      ⊢ Eq (AffineSubspace.map f (affineSpan k s)).direction (affineSpan k (Set.imag …
    -/
  · simp [direction_affineSpan]
    /-
      🎉 no goals
    -/
  · exact
      ⟨f p, mem_image_of_mem f (subset_affineSpan k _ hp),
        subset_affineSpan k _ (mem_image_of_mem f hp)⟩


/-- Affine map from a smaller to a larger subspace of the same space.

This is the affine version of `Submodule.inclusion`. -/
@[simps linear]
def inclusion (h : S₁ ≤ S₂) : S₁ →ᵃ[k] S₂ where
  toFun := Set.inclusion h
  linear := Submodule.inclusion <| AffineSubspace.direction_le h
  map_vadd' _ _ := rfl


@[simp]
theorem coe_inclusion_apply (h : S₁ ≤ S₂) (x : S₁) : (inclusion h x : P₁) = x :=
  rfl


@[simp]
theorem inclusion_rfl : inclusion (le_refl S₁) = AffineMap.id k S₁ := rfl


@[simp]
theorem map_top_of_surjective (hf : Function.Surjective f) : AffineSubspace.map f ⊤ = ⊤ := by
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    f : AffineMap k P₁ P₂
    hf : Function.Surjective ⇑f
    ⊢ Eq (AffineSubspace.map f Top.top) Top.top
  -/
  rw [AffineSubspace.ext_iff]
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    f : AffineMap k P₁ P₂
    hf : Function.Surjective ⇑f
    ⊢ Eq ↑(AffineSubspace.map f Top.top) ↑Top.top
  -/
  exact image_univ_of_surjective hf
  /-
    🎉 no goals
  -/


theorem span_eq_top_of_surjective {s : Set P₁} (hf : Function.Surjective f)
    (h : affineSpan k s = ⊤) : affineSpan k (f '' s) = ⊤ := by
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    f : AffineMap k P₁ P₂
    s : Set P₁
    hf : Function.Surjective ⇑f
    h : Eq (affineSpan k s) Top.top
    ⊢ Eq (affineSpan k (Set.image (⇑f) s)) Top.top
  -/
  rw [← AffineSubspace.map_span, h, map_top_of_surjective f hf]
  /-
    🎉 no goals
  -/


/-- Affine equivalence between two equal affine subspace.

This is the affine version of `LinearEquiv.ofEq`. -/
@[simps linear]
def ofEq (h : S₁ = S₂) : S₁ ≃ᵃ[k] S₂ where
  toEquiv := Equiv.Set.ofEq <| congr_arg _ h
  linear := .ofEq _ _ <| congr_arg _ h
  map_vadd' _ _ := rfl


@[simp]
theorem coe_ofEq_apply (h : S₁ = S₂) (x : S₁) : (ofEq S₁ S₂ h x : P₁) = x :=
  rfl


@[simp]
theorem ofEq_symm (h : S₁ = S₂) : (ofEq S₁ S₂ h).symm = ofEq S₂ S₁ h.symm :=
  rfl


@[simp]
theorem ofEq_rfl : ofEq S₁ S₁ rfl = AffineEquiv.refl k S₁ := rfl


theorem span_eq_top_iff {s : Set P₁} (e : P₁ ≃ᵃ[k] P₂) :
    affineSpan k s = ⊤ ↔ affineSpan k (e '' s) = ⊤ := by
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    s : Set P₁
    e : AffineEquiv k P₁ P₂
    ⊢ Iff (Eq (affineSpan k s) Top.top) (Eq (affineSpan k (Set.image (⇑e) s)) Top. …
  -/
  refine ⟨(e : P₁ →ᵃ[k] P₂).span_eq_top_of_surjective e.surjective, ?_⟩
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    s : Set P₁
    e : AffineEquiv k P₁ P₂
    ⊢ Eq (affineSpan k (Set.image (⇑e) s)) Top.top → Eq (affineSpan k s) Top.top
  -/
  intro h
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    s : Set P₁
    e : AffineEquiv k P₁ P₂
    h : Eq (affineSpan k (Set.image (⇑e) s)) Top.top
    ⊢ Eq (affineSpan k s) Top.top
  -/
  have : s = e.symm '' (e '' s) := by rw [← image_comp]; simp
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    s : Set P₁
    e : AffineEquiv k P₁ P₂
    h : Eq (affineSpan k (Set.image (⇑e) s)) Top.top
    this : Eq s (Set.image (⇑e.symm) (Set.image (⇑e) s))
    ⊢ Eq (affineSpan k s) Top.top
  -/
  rw [this]
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    s : Set P₁
    e : AffineEquiv k P₁ P₂
    h : Eq (affineSpan k (Set.image (⇑e) s)) Top.top
    this : Eq s (Set.image (⇑e.symm) (Set.image (⇑e) s))
    ⊢ Eq (affineSpan k (Set.image (⇑e.symm) (Set.image (⇑e) s))) Top.top
  -/
  exact (e.symm : P₂ →ᵃ[k] P₁).span_eq_top_of_surjective e.symm.surjective h
  /-
    🎉 no goals
  -/


/-- The preimage of an affine subspace under an affine map as an affine subspace. -/
def comap (f : P₁ →ᵃ[k] P₂) (s : AffineSubspace k P₂) : AffineSubspace k P₁ where
  carrier := f ⁻¹' s
  smul_vsub_vadd_mem t p₁ p₂ p₃ (hp₁ : f p₁ ∈ s) (hp₂ : f p₂ ∈ s) (hp₃ : f p₃ ∈ s) :=
    show f _ ∈ s by
      /-
        k : Type u_1
        V₁ : Type u_2
        P₁ : Type u_3
        V₂ : Type u_4
        P₂ : Type u_5
        V₃ : Type u_6
        P₃ : Type u_7
        inst✝⁹ : Ring k
        inst✝⁸ : AddCommGroup V₁
        inst✝⁷ : Module k V₁
        inst✝⁶ : AddTorsor V₁ P₁
        inst✝⁵ : AddCommGroup V₂
        inst✝⁴ : Module k V₂
        inst✝³ : AddTorsor V₂ P₂
        inst✝² : AddCommGroup V₃
        inst✝¹ : Module k V₃
        inst✝ : AddTorsor V₃ P₃
        f : AffineMap k P₁ P₂
        s : AffineSubspace k P₂
        t : k
        p₁ p₂ p₃ : P₁
        hp₁ : Membership.mem s (f p₁)
        hp₂ : Membership.mem s (f p₂)
        hp₃ : Membership.mem s (f p₃)
        ⊢ Membership.mem s (f (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub p₁ p₂)) p₃))
      -/
      rw [AffineMap.map_vadd, LinearMap.map_smul, AffineMap.linearMap_vsub]
      /-
        k : Type u_1
        V₁ : Type u_2
        P₁ : Type u_3
        V₂ : Type u_4
        P₂ : Type u_5
        V₃ : Type u_6
        P₃ : Type u_7
        inst✝⁹ : Ring k
        inst✝⁸ : AddCommGroup V₁
        inst✝⁷ : Module k V₁
        inst✝⁶ : AddTorsor V₁ P₁
        inst✝⁵ : AddCommGroup V₂
        inst✝⁴ : Module k V₂
        inst✝³ : AddTorsor V₂ P₂
        inst✝² : AddCommGroup V₃
        inst✝¹ : Module k V₃
        inst✝ : AddTorsor V₃ P₃
        f : AffineMap k P₁ P₂
        s : AffineSubspace k P₂
        t : k
        p₁ p₂ p₃ : P₁
        hp₁ : Membership.mem s (f p₁)
        hp₂ : Membership.mem s (f p₂)
        hp₃ : Membership.mem s (f p₃)
        ⊢ Membership.mem s (HVAdd.hVAdd (HSMul.hSMul t (VSub.vsub (f p₁) (f p₂))) (f p …
      -/
      apply s.smul_vsub_vadd_mem _ hp₁ hp₂ hp₃
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_comap (f : P₁ →ᵃ[k] P₂) (s : AffineSubspace k P₂) : (s.comap f : Set P₁) = f ⁻¹' ↑s :=
  rfl


@[simp]
theorem mem_comap {f : P₁ →ᵃ[k] P₂} {x : P₁} {s : AffineSubspace k P₂} : x ∈ s.comap f ↔ f x ∈ s :=
  Iff.rfl


theorem comap_mono {f : P₁ →ᵃ[k] P₂} {s t : AffineSubspace k P₂} : s ≤ t → s.comap f ≤ t.comap f :=
  preimage_mono


@[simp]
theorem comap_top {f : P₁ →ᵃ[k] P₂} : (⊤ : AffineSubspace k P₂).comap f = ⊤ := by
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    f : AffineMap k P₁ P₂
    ⊢ Eq (AffineSubspace.comap f Top.top) Top.top
  -/
  rw [AffineSubspace.ext_iff]
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    f : AffineMap k P₁ P₂
    ⊢ Eq ↑(AffineSubspace.comap f Top.top) ↑Top.top
  -/
  exact preimage_univ (f := f)
  /-
    🎉 no goals
  -/


@[simp] theorem comap_bot (f : P₁ →ᵃ[k] P₂) : comap f ⊥ = ⊥ := rfl


@[simp]
theorem comap_id (s : AffineSubspace k P₁) : s.comap (AffineMap.id k P₁) = s :=
  rfl


theorem comap_comap (s : AffineSubspace k P₃) (f : P₁ →ᵃ[k] P₂) (g : P₂ →ᵃ[k] P₃) :
    (s.comap g).comap f = s.comap (g.comp f) :=
  rfl

-- lemmas about map and comap derived from the galois connection

theorem map_le_iff_le_comap {f : P₁ →ᵃ[k] P₂} {s : AffineSubspace k P₁} {t : AffineSubspace k P₂} :
    s.map f ≤ t ↔ s ≤ t.comap f :=
  image_subset_iff


theorem gc_map_comap (f : P₁ →ᵃ[k] P₂) : GaloisConnection (map f) (comap f) := fun _ _ =>
  map_le_iff_le_comap


theorem map_comap_le (f : P₁ →ᵃ[k] P₂) (s : AffineSubspace k P₂) : (s.comap f).map f ≤ s :=
  (gc_map_comap f).l_u_le _


theorem le_comap_map (f : P₁ →ᵃ[k] P₂) (s : AffineSubspace k P₁) : s ≤ (s.map f).comap f :=
  (gc_map_comap f).le_u_l _


theorem map_sup (s t : AffineSubspace k P₁) (f : P₁ →ᵃ[k] P₂) : (s ⊔ t).map f = s.map f ⊔ t.map f :=
  (gc_map_comap f).l_sup


theorem map_iSup {ι : Sort*} (f : P₁ →ᵃ[k] P₂) (s : ι → AffineSubspace k P₁) :
    (iSup s).map f = ⨆ i, (s i).map f :=
  (gc_map_comap f).l_iSup


theorem comap_inf (s t : AffineSubspace k P₂) (f : P₁ →ᵃ[k] P₂) :
    (s ⊓ t).comap f = s.comap f ⊓ t.comap f :=
  (gc_map_comap f).u_inf


theorem comap_supr {ι : Sort*} (f : P₁ →ᵃ[k] P₂) (s : ι → AffineSubspace k P₂) :
    (iInf s).comap f = ⨅ i, (s i).comap f :=
  (gc_map_comap f).u_iInf


@[simp]
theorem comap_symm (e : P₁ ≃ᵃ[k] P₂) (s : AffineSubspace k P₁) :
    s.comap (e.symm : P₂ →ᵃ[k] P₁) = s.map e :=
  coe_injective <| e.preimage_symm _


@[simp]
theorem map_symm (e : P₁ ≃ᵃ[k] P₂) (s : AffineSubspace k P₂) :
    s.map (e.symm : P₂ →ᵃ[k] P₁) = s.comap e :=
  coe_injective <| e.image_symm _


theorem comap_span (f : P₁ ≃ᵃ[k] P₂) (s : Set P₂) :
    (affineSpan k s).comap (f : P₁ →ᵃ[k] P₂) = affineSpan k (f ⁻¹' s) := by
  /-
    k : Type u_1
    V₁ : Type u_2
    P₁ : Type u_3
    V₂ : Type u_4
    P₂ : Type u_5
    inst✝⁶ : Ring k
    inst✝⁵ : AddCommGroup V₁
    inst✝⁴ : Module k V₁
    inst✝³ : AddTorsor V₁ P₁
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module k V₂
    inst✝ : AddTorsor V₂ P₂
    f : AffineEquiv k P₁ P₂
    s : Set P₂
    ⊢ Eq (AffineSubspace.comap (↑f) (affineSpan k s)) (affineSpan k (Set.preimage  …
  -/
  rw [← map_symm, map_span, AffineEquiv.coe_coe, f.image_symm]
  /-
    🎉 no goals
  -/


/-- Two affine subspaces are parallel if one is related to the other by adding the same vector
to all points. -/
def Parallel (s₁ s₂ : AffineSubspace k P) : Prop :=
  ∃ v : V, s₂ = s₁.map (constVAdd k P v)


@[inherit_doc]
scoped[Affine] infixl:50 " ∥ " => AffineSubspace.Parallel


@[symm]
theorem Parallel.symm {s₁ s₂ : AffineSubspace k P} (h : s₁ ∥ s₂) : s₂ ∥ s₁ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s₁ s₂ : AffineSubspace k P
    h : s₁.Parallel s₂
    ⊢ s₂.Parallel s₁
  -/
  rcases h with ⟨v, rfl⟩
  /-
    case intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s₁ : AffineSubspace k P
    v : V
    ⊢ (AffineSubspace.map (↑(AffineEquiv.constVAdd k P v)) s₁).Parallel s₁
  -/
  refine ⟨-v, ?_⟩
  rw [map_map, ← coe_trans_to_affineMap, ← constVAdd_add, neg_add_cancel, constVAdd_zero,
    coe_refl_to_affineMap, map_id]


theorem parallel_comm {s₁ s₂ : AffineSubspace k P} : s₁ ∥ s₂ ↔ s₂ ∥ s₁ :=
  ⟨Parallel.symm, Parallel.symm⟩


@[refl]
theorem Parallel.refl (s : AffineSubspace k P) : s ∥ s :=
         /-
           k : Type u_1
           V : Type u_2
           P : Type u_3
           inst✝³ : Ring k
           inst✝² : AddCommGroup V
           inst✝¹ : Module k V
           inst✝ : AddTorsor V P
           s : AffineSubspace k P
           ⊢ Eq s (AffineSubspace.map (↑(AffineEquiv.constVAdd k P 0)) s)
         -/
  ⟨0, by simp⟩
         /-
           🎉 no goals
         -/


@[trans]
theorem Parallel.trans {s₁ s₂ s₃ : AffineSubspace k P} (h₁₂ : s₁ ∥ s₂) (h₂₃ : s₂ ∥ s₃) :
    s₁ ∥ s₃ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s₁ s₂ s₃ : AffineSubspace k P
    h₁₂ : s₁.Parallel s₂
    h₂₃ : s₂.Parallel s₃
    ⊢ s₁.Parallel s₃
  -/
  rcases h₁₂ with ⟨v₁₂, rfl⟩
  /-
    case intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s₁ s₃ : AffineSubspace k P
    v₁₂ : V
    h₂₃ : (AffineSubspace.map (↑(AffineEquiv.constVAdd k P v₁₂)) s₁).Parallel s₃
    ⊢ s₁.Parallel s₃
  -/
  rcases h₂₃ with ⟨v₂₃, rfl⟩
  /-
    case intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s₁ : AffineSubspace k P
    v₁₂ v₂₃ : V
    ⊢ s₁.Parallel (AffineSubspace.map (↑(AffineEquiv.constVAdd k P v₂₃)) (AffineSu …
  -/
  refine ⟨v₂₃ + v₁₂, ?_⟩
  /-
    case intro.intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s₁ : AffineSubspace k P
    v₁₂ v₂₃ : V
    ⊢ Eq (AffineSubspace.map (↑(AffineEquiv.constVAdd k P v₂₃)) (AffineSubspace.ma …
  -/
  rw [map_map, ← coe_trans_to_affineMap, ← constVAdd_add]
  /-
    🎉 no goals
  -/


theorem Parallel.direction_eq {s₁ s₂ : AffineSubspace k P} (h : s₁ ∥ s₂) :
    s₁.direction = s₂.direction := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s₁ s₂ : AffineSubspace k P
    h : s₁.Parallel s₂
    ⊢ Eq s₁.direction s₂.direction
  -/
  rcases h with ⟨v, rfl⟩
  /-
    case intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s₁ : AffineSubspace k P
    v : V
    ⊢ Eq s₁.direction (AffineSubspace.map (↑(AffineEquiv.constVAdd k P v)) s₁).dir …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem parallel_bot_iff_eq_bot {s : AffineSubspace k P} : s ∥ ⊥ ↔ s = ⊥ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    ⊢ Iff (s.Parallel Bot.bot) (Eq s Bot.bot)
  -/
  refine ⟨fun h => ?_, fun h => h ▸ Parallel.refl _⟩
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    h : s.Parallel Bot.bot
    ⊢ Eq s Bot.bot
  -/
  rcases h with ⟨v, h⟩
  /-
    case intro
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    v : V
    h : Eq Bot.bot (AffineSubspace.map (↑(AffineEquiv.constVAdd k P v)) s)
    ⊢ Eq s Bot.bot
  -/
  rwa [eq_comm, map_eq_bot_iff] at h
  /-
    🎉 no goals
  -/


@[simp]
theorem bot_parallel_iff_eq_bot {s : AffineSubspace k P} : ⊥ ∥ s ↔ s = ⊥ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s : AffineSubspace k P
    ⊢ Iff (Bot.bot.Parallel s) (Eq s Bot.bot)
  -/
  rw [parallel_comm, parallel_bot_iff_eq_bot]
  /-
    🎉 no goals
  -/


theorem parallel_iff_direction_eq_and_eq_bot_iff_eq_bot {s₁ s₂ : AffineSubspace k P} :
    s₁ ∥ s₂ ↔ s₁.direction = s₂.direction ∧ (s₁ = ⊥ ↔ s₂ = ⊥) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s₁ s₂ : AffineSubspace k P
    ⊢ Iff (s₁.Parallel s₂) (And (Eq s₁.direction s₂.direction) (Iff (Eq s₁ Bot.bot …
  -/
  refine ⟨fun h => ⟨h.direction_eq, ?_, ?_⟩, fun h => ?_⟩
    /-
      case refine_1
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s₁ s₂ : AffineSubspace k P
      h : s₁.Parallel s₂
      ⊢ Eq s₁ Bot.bot → Eq s₂ Bot.bot
    -/
  · rintro rfl
    /-
      case refine_1
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s₂ : AffineSubspace k P
      h : Bot.bot.Parallel s₂
      ⊢ Eq s₂ Bot.bot
    -/
    exact bot_parallel_iff_eq_bot.1 h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s₁ s₂ : AffineSubspace k P
      h : s₁.Parallel s₂
      ⊢ Eq s₂ Bot.bot → Eq s₁ Bot.bot
    -/
  · rintro rfl
    /-
      case refine_2
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s₁ : AffineSubspace k P
      h : s₁.Parallel Bot.bot
      ⊢ Eq s₁ Bot.bot
    -/
    exact parallel_bot_iff_eq_bot.1 h
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s₁ s₂ : AffineSubspace k P
      h : And (Eq s₁.direction s₂.direction) (Iff (Eq s₁ Bot.bot) (Eq s₂ Bot.bot))
      ⊢ s₁.Parallel s₂
    -/
  · rcases h with ⟨hd, hb⟩
    /-
      case refine_3.intro
      k : Type u_1
      V : Type u_2
      P : Type u_3
      inst✝³ : Ring k
      inst✝² : AddCommGroup V
      inst✝¹ : Module k V
      inst✝ : AddTorsor V P
      s₁ s₂ : AffineSubspace k P
      hd : Eq s₁.direction s₂.direction
      hb : Iff (Eq s₁ Bot.bot) (Eq s₂ Bot.bot)
      ⊢ s₁.Parallel s₂
    -/
    by_cases hs₁ : s₁ = ⊥
      /-
        case pos
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : Ring k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s₁ s₂ : AffineSubspace k P
        hd : Eq s₁.direction s₂.direction
        hb : Iff (Eq s₁ Bot.bot) (Eq s₂ Bot.bot)
        hs₁ : Eq s₁ Bot.bot
        ⊢ s₁.Parallel s₂
      -/
    · rw [hs₁, bot_parallel_iff_eq_bot]
      /-
        case pos
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : Ring k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s₁ s₂ : AffineSubspace k P
        hd : Eq s₁.direction s₂.direction
        hb : Iff (Eq s₁ Bot.bot) (Eq s₂ Bot.bot)
        hs₁ : Eq s₁ Bot.bot
        ⊢ Eq s₂ Bot.bot
      -/
      exact hb.1 hs₁
      /-
        🎉 no goals
      -/
      /-
        case neg
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : Ring k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s₁ s₂ : AffineSubspace k P
        hd : Eq s₁.direction s₂.direction
        hb : Iff (Eq s₁ Bot.bot) (Eq s₂ Bot.bot)
        hs₁ : Not (Eq s₁ Bot.bot)
        ⊢ s₁.Parallel s₂
      -/
    · have hs₂ : s₂ ≠ ⊥ := hb.not.1 hs₁
      /-
        case neg
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : Ring k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s₁ s₂ : AffineSubspace k P
        hd : Eq s₁.direction s₂.direction
        hb : Iff (Eq s₁ Bot.bot) (Eq s₂ Bot.bot)
        hs₁ : Not (Eq s₁ Bot.bot)
        hs₂ : Ne s₂ Bot.bot
        ⊢ s₁.Parallel s₂
      -/
      rcases (nonempty_iff_ne_bot s₁).2 hs₁ with ⟨p₁, hp₁⟩
      /-
        case neg.intro
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : Ring k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s₁ s₂ : AffineSubspace k P
        hd : Eq s₁.direction s₂.direction
        hb : Iff (Eq s₁ Bot.bot) (Eq s₂ Bot.bot)
        hs₁ : Not (Eq s₁ Bot.bot)
        hs₂ : Ne s₂ Bot.bot
        p₁ : P
        hp₁ : Membership.mem (↑s₁) p₁
        ⊢ s₁.Parallel s₂
      -/
      rcases (nonempty_iff_ne_bot s₂).2 hs₂ with ⟨p₂, hp₂⟩
      /-
        case neg.intro.intro
        k : Type u_1
        V : Type u_2
        P : Type u_3
        inst✝³ : Ring k
        inst✝² : AddCommGroup V
        inst✝¹ : Module k V
        inst✝ : AddTorsor V P
        s₁ s₂ : AffineSubspace k P
        hd : Eq s₁.direction s₂.direction
        hb : Iff (Eq s₁ Bot.bot) (Eq s₂ Bot.bot)
        hs₁ : Not (Eq s₁ Bot.bot)
        hs₂ : Ne s₂ Bot.bot
        p₁ : P
        hp₁ : Membership.mem (↑s₁) p₁
        p₂ : P
        hp₂ : Membership.mem (↑s₂) p₂
        ⊢ s₁.Parallel s₂
      -/
      refine ⟨p₂ -ᵥ p₁, (eq_iff_direction_eq_of_mem hp₂ ?_).2 ?_⟩
        /-
          case neg.intro.intro.refine_1
          k : Type u_1
          V : Type u_2
          P : Type u_3
          inst✝³ : Ring k
          inst✝² : AddCommGroup V
          inst✝¹ : Module k V
          inst✝ : AddTorsor V P
          s₁ s₂ : AffineSubspace k P
          hd : Eq s₁.direction s₂.direction
          hb : Iff (Eq s₁ Bot.bot) (Eq s₂ Bot.bot)
          hs₁ : Not (Eq s₁ Bot.bot)
          hs₂ : Ne s₂ Bot.bot
          p₁ : P
          hp₁ : Membership.mem (↑s₁) p₁
          p₂ : P
          hp₂ : Membership.mem (↑s₂) p₂
          ⊢ Membership.mem (AffineSubspace.map (↑(AffineEquiv.constVAdd k P (VSub.vsub p …
        -/
      · rw [mem_map]
        /-
          case neg.intro.intro.refine_1
          k : Type u_1
          V : Type u_2
          P : Type u_3
          inst✝³ : Ring k
          inst✝² : AddCommGroup V
          inst✝¹ : Module k V
          inst✝ : AddTorsor V P
          s₁ s₂ : AffineSubspace k P
          hd : Eq s₁.direction s₂.direction
          hb : Iff (Eq s₁ Bot.bot) (Eq s₂ Bot.bot)
          hs₁ : Not (Eq s₁ Bot.bot)
          hs₂ : Ne s₂ Bot.bot
          p₁ : P
          hp₁ : Membership.mem (↑s₁) p₁
          p₂ : P
          hp₂ : Membership.mem (↑s₂) p₂
          ⊢ Exists fun y => And (Membership.mem s₁ y) (Eq (↑(AffineEquiv.constVAdd k P ( …
        -/
        refine ⟨p₁, hp₁, ?_⟩
        /-
          case neg.intro.intro.refine_1
          k : Type u_1
          V : Type u_2
          P : Type u_3
          inst✝³ : Ring k
          inst✝² : AddCommGroup V
          inst✝¹ : Module k V
          inst✝ : AddTorsor V P
          s₁ s₂ : AffineSubspace k P
          hd : Eq s₁.direction s₂.direction
          hb : Iff (Eq s₁ Bot.bot) (Eq s₂ Bot.bot)
          hs₁ : Not (Eq s₁ Bot.bot)
          hs₂ : Ne s₂ Bot.bot
          p₁ : P
          hp₁ : Membership.mem (↑s₁) p₁
          p₂ : P
          hp₂ : Membership.mem (↑s₂) p₂
          ⊢ Eq (↑(AffineEquiv.constVAdd k P (VSub.vsub p₂ p₁)) p₁) p₂
        -/
        simp
        /-
          🎉 no goals
        -/
        /-
          case neg.intro.intro.refine_2
          k : Type u_1
          V : Type u_2
          P : Type u_3
          inst✝³ : Ring k
          inst✝² : AddCommGroup V
          inst✝¹ : Module k V
          inst✝ : AddTorsor V P
          s₁ s₂ : AffineSubspace k P
          hd : Eq s₁.direction s₂.direction
          hb : Iff (Eq s₁ Bot.bot) (Eq s₂ Bot.bot)
          hs₁ : Not (Eq s₁ Bot.bot)
          hs₂ : Ne s₂ Bot.bot
          p₁ : P
          hp₁ : Membership.mem (↑s₁) p₁
          p₂ : P
          hp₂ : Membership.mem (↑s₂) p₂
          ⊢ Eq s₂.direction (AffineSubspace.map (↑(AffineEquiv.constVAdd k P (VSub.vsub  …
        -/
      · simpa using hd.symm
        /-
          🎉 no goals
        -/


theorem Parallel.vectorSpan_eq {s₁ s₂ : Set P} (h : affineSpan k s₁ ∥ affineSpan k s₂) :
    vectorSpan k s₁ = vectorSpan k s₂ := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s₁ s₂ : Set P
    h : (affineSpan k s₁).Parallel (affineSpan k s₂)
    ⊢ Eq (vectorSpan k s₁) (vectorSpan k s₂)
  -/
  simp_rw [← direction_affineSpan]
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s₁ s₂ : Set P
    h : (affineSpan k s₁).Parallel (affineSpan k s₂)
    ⊢ Eq (affineSpan k s₁).direction (affineSpan k s₂).direction
  -/
  exact h.direction_eq
  /-
    🎉 no goals
  -/


theorem affineSpan_parallel_iff_vectorSpan_eq_and_eq_empty_iff_eq_empty {s₁ s₂ : Set P} :
    affineSpan k s₁ ∥ affineSpan k s₂ ↔ vectorSpan k s₁ = vectorSpan k s₂ ∧ (s₁ = ∅ ↔ s₂ = ∅) := by
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s₁ s₂ : Set P
    ⊢ Iff ((affineSpan k s₁).Parallel (affineSpan k s₂)) (And (Eq (vectorSpan k s₁ …
  -/
  repeat rw [← direction_affineSpan, ← affineSpan_eq_bot k]
  -- Porting note: more issues with `simp`
  /-
    k : Type u_1
    V : Type u_2
    P : Type u_3
    inst✝³ : Ring k
    inst✝² : AddCommGroup V
    inst✝¹ : Module k V
    inst✝ : AddTorsor V P
    s₁ s₂ : Set P
    ⊢ Iff ((affineSpan k s₁).Parallel (affineSpan k s₂)) (And (Eq (affineSpan k s₁ …
  -/
  exact parallel_iff_direction_eq_and_eq_bot_iff_eq_bot
  /-
    🎉 no goals
  -/


theorem affineSpan_pair_parallel_iff_vectorSpan_eq {p₁ p₂ p₃ p₄ : P} :
    line[k, p₁, p₂] ∥ line[k, p₃, p₄] ↔
      vectorSpan k ({p₁, p₂} : Set P) = vectorSpan k ({p₃, p₄} : Set P) := by
  simp [affineSpan_parallel_iff_vectorSpan_eq_and_eq_empty_iff_eq_empty, ←
    not_nonempty_iff_eq_empty]


