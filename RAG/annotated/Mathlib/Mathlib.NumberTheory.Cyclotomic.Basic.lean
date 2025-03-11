/-- Given an `A`-algebra `B` and `S : Set ℕ+`, we define `IsCyclotomicExtension S A B` requiring
that there is an `n`-th primitive root of unity in `B` for all `n ∈ S` and that `B` is generated
over `A` by the roots of `X ^ n - 1`. -/

@[mk_iff]
class IsCyclotomicExtension : Prop where
  /-- For all `n ∈ S`, there exists a primitive `n`-th root of unity in `B`. -/
  exists_prim_root {n : ℕ+} (ha : n ∈ S) : ∃ r : B, IsPrimitiveRoot r n
  /-- The `n`-th roots of unity, for `n ∈ S`, generate `B` as an `A`-algebra. -/
  adjoin_roots : ∀ x : B, x ∈ adjoin A {b : B | ∃ n : ℕ+, n ∈ S ∧ b ^ (n : ℕ) = 1}


/-- A reformulation of `IsCyclotomicExtension` that uses `⊤`. -/
theorem iff_adjoin_eq_top :
    IsCyclotomicExtension S A B ↔
      (∀ n : ℕ+, n ∈ S → ∃ r : B, IsPrimitiveRoot r n) ∧
        adjoin A {b : B | ∃ n : ℕ+, n ∈ S ∧ b ^ (n : ℕ) = 1} = ⊤ :=
  ⟨fun h => ⟨fun _ => h.exists_prim_root, Algebra.eq_top_iff.2 h.adjoin_roots⟩, fun h =>
    ⟨h.1 _, Algebra.eq_top_iff.1 h.2⟩⟩


/-- A reformulation of `IsCyclotomicExtension` in the case `S` is a singleton. -/
theorem iff_singleton :
    IsCyclotomicExtension {n} A B ↔
      (∃ r : B, IsPrimitiveRoot r n) ∧ ∀ x, x ∈ adjoin A {b : B | b ^ (n : ℕ) = 1} := by
  /-
    n : PNat
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    ⊢ Iff (IsCyclotomicExtension (Singleton.singleton n) A B) (And (Exists fun r = …
  -/
  simp [isCyclotomicExtension_iff]
  /-
    🎉 no goals
  -/


/-- If `IsCyclotomicExtension ∅ A B`, then the image of `A` in `B` equals `B`. -/
theorem empty [h : IsCyclotomicExtension ∅ A B] : (⊥ : Subalgebra A B) = ⊤ := by
  /-
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    h : IsCyclotomicExtension EmptyCollection.emptyCollection A B
    ⊢ Eq Bot.bot Top.top
  -/
  simpa [Algebra.eq_top_iff, isCyclotomicExtension_iff] using h
  /-
    🎉 no goals
  -/


/-- If `IsCyclotomicExtension {1} A B`, then the image of `A` in `B` equals `B`. -/
theorem singleton_one [h : IsCyclotomicExtension {1} A B] : (⊥ : Subalgebra A B) = ⊤ :=
  Algebra.eq_top_iff.2 fun x => by
    /-
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : IsCyclotomicExtension (Singleton.singleton 1) A B
      x : B
      ⊢ Membership.mem Bot.bot x
    -/
    simpa [adjoin_singleton_one] using ((isCyclotomicExtension_iff _ _ _).1 h).2 x
    /-
      🎉 no goals
    -/


/-- If `(⊥ : SubAlgebra A B) = ⊤`, then `IsCyclotomicExtension ∅ A B`. -/
theorem singleton_zero_of_bot_eq_top (h : (⊥ : Subalgebra A B) = ⊤) :
    IsCyclotomicExtension ∅ A B := by
-- Porting note: Lean3 is able to infer `A`.
  refine (iff_adjoin_eq_top _ A _).2
    ⟨fun s hs => by simp at hs, _root_.eq_top_iff.2 fun x hx => ?_⟩
  /-
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    h : Eq Bot.bot Top.top
    x : B
    hx : Membership.mem Top.top x
    ⊢ Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n => And (Member …
  -/
  rw [← h] at hx
  /-
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    h : Eq Bot.bot Top.top
    x : B
    hx : Membership.mem Bot.bot x
    ⊢ Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n => And (Member …
  -/
  simpa using hx
  /-
    🎉 no goals
  -/


/-- Transitivity of cyclotomic extensions. -/
theorem trans (C : Type w) [CommRing C] [Algebra A C] [Algebra B C] [IsScalarTower A B C]
    [hS : IsCyclotomicExtension S A B] [hT : IsCyclotomicExtension T B C]
    (h : Function.Injective (algebraMap B C)) : IsCyclotomicExtension (S ∪ T) A C := by
  /-
    S T : Set PNat
    A : Type u
    B : Type v
    inst✝⁶ : CommRing A
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra A B
    C : Type w
    inst✝³ : CommRing C
    inst✝² : Algebra A C
    inst✝¹ : Algebra B C
    inst✝ : IsScalarTower A B C
    hS : IsCyclotomicExtension S A B
    hT : IsCyclotomicExtension T B C
    h : Function.Injective ⇑(algebraMap B C)
    ⊢ IsCyclotomicExtension (Union.union S T) A C
  -/
  refine ⟨fun hn => ?_, fun x => ?_⟩
    /-
      case refine_1
      S T : Set PNat
      A : Type u
      B : Type v
      inst✝⁶ : CommRing A
      inst✝⁵ : CommRing B
      inst✝⁴ : Algebra A B
      C : Type w
      inst✝³ : CommRing C
      inst✝² : Algebra A C
      inst✝¹ : Algebra B C
      inst✝ : IsScalarTower A B C
      hS : IsCyclotomicExtension S A B
      hT : IsCyclotomicExtension T B C
      h : Function.Injective ⇑(algebraMap B C)
      n✝ : PNat
      hn : Membership.mem (Union.union S T) n✝
      ⊢ Exists fun r => IsPrimitiveRoot r ↑n✝
    -/
  · cases' hn with hn hn
      /-
        case refine_1.inl
        S T : Set PNat
        A : Type u
        B : Type v
        inst✝⁶ : CommRing A
        inst✝⁵ : CommRing B
        inst✝⁴ : Algebra A B
        C : Type w
        inst✝³ : CommRing C
        inst✝² : Algebra A C
        inst✝¹ : Algebra B C
        inst✝ : IsScalarTower A B C
        hS : IsCyclotomicExtension S A B
        hT : IsCyclotomicExtension T B C
        h : Function.Injective ⇑(algebraMap B C)
        n✝ : PNat
        hn : Membership.mem S n✝
        ⊢ Exists fun r => IsPrimitiveRoot r ↑n✝
      -/
    · obtain ⟨b, hb⟩ := ((isCyclotomicExtension_iff _ _ _).1 hS).1 hn
      /-
        case refine_1.inl.intro
        S T : Set PNat
        A : Type u
        B : Type v
        inst✝⁶ : CommRing A
        inst✝⁵ : CommRing B
        inst✝⁴ : Algebra A B
        C : Type w
        inst✝³ : CommRing C
        inst✝² : Algebra A C
        inst✝¹ : Algebra B C
        inst✝ : IsScalarTower A B C
        hS : IsCyclotomicExtension S A B
        hT : IsCyclotomicExtension T B C
        h : Function.Injective ⇑(algebraMap B C)
        n✝ : PNat
        hn : Membership.mem S n✝
        b : B
        hb : IsPrimitiveRoot b ↑n✝
        ⊢ Exists fun r => IsPrimitiveRoot r ↑n✝
      -/
      refine ⟨algebraMap B C b, ?_⟩
      /-
        case refine_1.inl.intro
        S T : Set PNat
        A : Type u
        B : Type v
        inst✝⁶ : CommRing A
        inst✝⁵ : CommRing B
        inst✝⁴ : Algebra A B
        C : Type w
        inst✝³ : CommRing C
        inst✝² : Algebra A C
        inst✝¹ : Algebra B C
        inst✝ : IsScalarTower A B C
        hS : IsCyclotomicExtension S A B
        hT : IsCyclotomicExtension T B C
        h : Function.Injective ⇑(algebraMap B C)
        n✝ : PNat
        hn : Membership.mem S n✝
        b : B
        hb : IsPrimitiveRoot b ↑n✝
        ⊢ IsPrimitiveRoot ((algebraMap B C) b) ↑n✝
      -/
      exact hb.map_of_injective h
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr
        S T : Set PNat
        A : Type u
        B : Type v
        inst✝⁶ : CommRing A
        inst✝⁵ : CommRing B
        inst✝⁴ : Algebra A B
        C : Type w
        inst✝³ : CommRing C
        inst✝² : Algebra A C
        inst✝¹ : Algebra B C
        inst✝ : IsScalarTower A B C
        hS : IsCyclotomicExtension S A B
        hT : IsCyclotomicExtension T B C
        h : Function.Injective ⇑(algebraMap B C)
        n✝ : PNat
        hn : Membership.mem T n✝
        ⊢ Exists fun r => IsPrimitiveRoot r ↑n✝
      -/
    · exact ((isCyclotomicExtension_iff _ _ _).1 hT).1 hn
      /-
        🎉 no goals
      -/
  · refine adjoin_induction (hx := ((isCyclotomicExtension_iff T B _).1 hT).2 x)
      (fun c ⟨n, hn⟩ => subset_adjoin ⟨n, Or.inr hn.1, hn.2⟩) (fun b => ?_)
      (fun x y _ _ hx hy => Subalgebra.add_mem _ hx hy)
      fun x y _ _ hx hy => Subalgebra.mul_mem _ hx hy
    /-
      case refine_2
      S T : Set PNat
      A : Type u
      B : Type v
      inst✝⁶ : CommRing A
      inst✝⁵ : CommRing B
      inst✝⁴ : Algebra A B
      C : Type w
      inst✝³ : CommRing C
      inst✝² : Algebra A C
      inst✝¹ : Algebra B C
      inst✝ : IsScalarTower A B C
      hS : IsCyclotomicExtension S A B
      hT : IsCyclotomicExtension T B C
      h : Function.Injective ⇑(algebraMap B C)
      x : C
      b : B
      ⊢ Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n => And (Member …
    -/
    let f := IsScalarTower.toAlgHom A B C
    have hb : f b ∈ (adjoin A {b : B | ∃ a : ℕ+, a ∈ S ∧ b ^ (a : ℕ) = 1}).map f :=
      ⟨b, ((isCyclotomicExtension_iff _ _ _).1 hS).2 b, rfl⟩
    /-
      case refine_2
      S T : Set PNat
      A : Type u
      B : Type v
      inst✝⁶ : CommRing A
      inst✝⁵ : CommRing B
      inst✝⁴ : Algebra A B
      C : Type w
      inst✝³ : CommRing C
      inst✝² : Algebra A C
      inst✝¹ : Algebra B C
      inst✝ : IsScalarTower A B C
      hS : IsCyclotomicExtension S A B
      hT : IsCyclotomicExtension T B C
      h : Function.Injective ⇑(algebraMap B C)
      x : C
      b : B
      f : AlgHom A B C := IsScalarTower.toAlgHom A B C
      hb : Membership.mem (Subalgebra.map f (Algebra.adjoin A (setOf fun b => Exists …
      ⊢ Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n => And (Member …
    -/
    rw [IsScalarTower.toAlgHom_apply, ← adjoin_image] at hb
    /-
      case refine_2
      S T : Set PNat
      A : Type u
      B : Type v
      inst✝⁶ : CommRing A
      inst✝⁵ : CommRing B
      inst✝⁴ : Algebra A B
      C : Type w
      inst✝³ : CommRing C
      inst✝² : Algebra A C
      inst✝¹ : Algebra B C
      inst✝ : IsScalarTower A B C
      hS : IsCyclotomicExtension S A B
      hT : IsCyclotomicExtension T B C
      h : Function.Injective ⇑(algebraMap B C)
      x : C
      b : B
      f : AlgHom A B C := IsScalarTower.toAlgHom A B C
      hb : Membership.mem (Algebra.adjoin A (Set.image (⇑f) (setOf fun b => Exists f …
      ⊢ Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n => And (Member …
    -/
    refine adjoin_mono (fun y hy => ?_) hb
    /-
      case refine_2
      S T : Set PNat
      A : Type u
      B : Type v
      inst✝⁶ : CommRing A
      inst✝⁵ : CommRing B
      inst✝⁴ : Algebra A B
      C : Type w
      inst✝³ : CommRing C
      inst✝² : Algebra A C
      inst✝¹ : Algebra B C
      inst✝ : IsScalarTower A B C
      hS : IsCyclotomicExtension S A B
      hT : IsCyclotomicExtension T B C
      h : Function.Injective ⇑(algebraMap B C)
      x : C
      b : B
      f : AlgHom A B C := IsScalarTower.toAlgHom A B C
      hb : Membership.mem (Algebra.adjoin A (Set.image (⇑f) (setOf fun b => Exists f …
      y : C
      hy : Membership.mem (Set.image (⇑f) (setOf fun b => Exists fun a => And (Membe …
      ⊢ Membership.mem (setOf fun b => Exists fun n => And (Membership.mem (Union.un …
    -/
    obtain ⟨b₁, ⟨⟨n, hn⟩, h₁⟩⟩ := hy
    /-
      case refine_2.intro.intro.intro
      S T : Set PNat
      A : Type u
      B : Type v
      inst✝⁶ : CommRing A
      inst✝⁵ : CommRing B
      inst✝⁴ : Algebra A B
      C : Type w
      inst✝³ : CommRing C
      inst✝² : Algebra A C
      inst✝¹ : Algebra B C
      inst✝ : IsScalarTower A B C
      hS : IsCyclotomicExtension S A B
      hT : IsCyclotomicExtension T B C
      h : Function.Injective ⇑(algebraMap B C)
      x : C
      b : B
      f : AlgHom A B C := IsScalarTower.toAlgHom A B C
      hb : Membership.mem (Algebra.adjoin A (Set.image (⇑f) (setOf fun b => Exists f …
      y : C
      b₁ : B
      h₁ : Eq (f b₁) y
      n : PNat
      hn : And (Membership.mem S n) (Eq (HPow.hPow b₁ ↑n) 1)
      ⊢ Membership.mem (setOf fun b => Exists fun n => And (Membership.mem (Union.un …
    -/
    exact ⟨n, ⟨mem_union_left T hn.1, by rw [← h₁, ← map_pow, hn.2, map_one]⟩⟩
    /-
      🎉 no goals
    -/


@[nontriviality]
theorem subsingleton_iff [Subsingleton B] : IsCyclotomicExtension S A B ↔ S = { } ∨ S = {1} := by
  /-
    S : Set PNat
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : Subsingleton B
    ⊢ Iff (IsCyclotomicExtension S A B) (Or (Eq S EmptyCollection.emptyCollection) …
  -/
  have : Subsingleton (Subalgebra A B) := inferInstance
  /-
    S : Set PNat
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : Subsingleton B
    this : Subsingleton (Subalgebra A B)
    ⊢ Iff (IsCyclotomicExtension S A B) (Or (Eq S EmptyCollection.emptyCollection) …
  -/
  constructor
    /-
      case mp
      S : Set PNat
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : Subsingleton B
      this : Subsingleton (Subalgebra A B)
      ⊢ IsCyclotomicExtension S A B → Or (Eq S EmptyCollection.emptyCollection) (Eq  …
    -/
  · rintro ⟨hprim, -⟩
    /-
      case mp.mk
      S : Set PNat
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : Subsingleton B
      this : Subsingleton (Subalgebra A B)
      hprim : ∀ {n : PNat}, Membership.mem S n → Exists fun r => IsPrimitiveRoot r ↑n
      ⊢ Or (Eq S EmptyCollection.emptyCollection) (Eq S (Singleton.singleton 1))
    -/
    rw [← subset_singleton_iff_eq]
    /-
      case mp.mk
      S : Set PNat
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : Subsingleton B
      this : Subsingleton (Subalgebra A B)
      hprim : ∀ {n : PNat}, Membership.mem S n → Exists fun r => IsPrimitiveRoot r ↑n
      ⊢ HasSubset.Subset S (Singleton.singleton 1)
    -/
    intro t ht
    /-
      case mp.mk
      S : Set PNat
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : Subsingleton B
      this : Subsingleton (Subalgebra A B)
      hprim : ∀ {n : PNat}, Membership.mem S n → Exists fun r => IsPrimitiveRoot r ↑n
      t : PNat
      ht : Membership.mem S t
      ⊢ Membership.mem (Singleton.singleton 1) t
    -/
    obtain ⟨ζ, hζ⟩ := hprim ht
    /-
      case mp.mk.intro
      S : Set PNat
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : Subsingleton B
      this : Subsingleton (Subalgebra A B)
      hprim : ∀ {n : PNat}, Membership.mem S n → Exists fun r => IsPrimitiveRoot r ↑n
      t : PNat
      ht : Membership.mem S t
      ζ : B
      hζ : IsPrimitiveRoot ζ ↑t
      ⊢ Membership.mem (Singleton.singleton 1) t
    -/
    rw [mem_singleton_iff, ← PNat.coe_eq_one_iff]
    /-
      case mp.mk.intro
      S : Set PNat
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : Subsingleton B
      this : Subsingleton (Subalgebra A B)
      hprim : ∀ {n : PNat}, Membership.mem S n → Exists fun r => IsPrimitiveRoot r ↑n
      t : PNat
      ht : Membership.mem S t
      ζ : B
      hζ : IsPrimitiveRoot ζ ↑t
      ⊢ Eq (↑t) 1
    -/
    exact mod_cast hζ.unique (IsPrimitiveRoot.of_subsingleton ζ)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      S : Set PNat
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : Subsingleton B
      this : Subsingleton (Subalgebra A B)
      ⊢ Or (Eq S EmptyCollection.emptyCollection) (Eq S (Singleton.singleton 1)) → I …
    -/
  · rintro (rfl | rfl)
-- Porting note: `R := A` was not needed.
      /-
        case mpr.inl
        A : Type u
        B : Type v
        inst✝³ : CommRing A
        inst✝² : CommRing B
        inst✝¹ : Algebra A B
        inst✝ : Subsingleton B
        this : Subsingleton (Subalgebra A B)
        ⊢ IsCyclotomicExtension EmptyCollection.emptyCollection A B
      -/
    · exact ⟨fun h => h.elim, fun x => by convert (mem_top (R := A) : x ∈ ⊤)⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        A : Type u
        B : Type v
        inst✝³ : CommRing A
        inst✝² : CommRing B
        inst✝¹ : Algebra A B
        inst✝ : Subsingleton B
        this : Subsingleton (Subalgebra A B)
        ⊢ IsCyclotomicExtension (Singleton.singleton 1) A B
      -/
    · rw [iff_singleton]
      exact ⟨⟨0, IsPrimitiveRoot.of_subsingleton 0⟩,
        fun x => by convert (mem_top (R := A) : x ∈ ⊤)⟩


/-- If `B` is a cyclotomic extension of `A` given by roots of unity of order in `S ∪ T`, then `B`
is a cyclotomic extension of `adjoin A { b : B | ∃ a : ℕ+, a ∈ S ∧ b ^ (a : ℕ) = 1 }` given by
roots of unity of order in `T`. -/
theorem union_right [h : IsCyclotomicExtension (S ∪ T) A B] :
    IsCyclotomicExtension T (adjoin A {b : B | ∃ a : ℕ+, a ∈ S ∧ b ^ (a : ℕ) = 1}) B := by
  have : {b : B | ∃ n : ℕ+, n ∈ S ∪ T ∧ b ^ (n : ℕ) = 1} =
      {b : B | ∃ n : ℕ+, n ∈ S ∧ b ^ (n : ℕ) = 1} ∪
        {b : B | ∃ n : ℕ+, n ∈ T ∧ b ^ (n : ℕ) = 1} := by
    refine le_antisymm ?_ ?_
    · rintro x ⟨n, hn₁ | hn₂, hnpow⟩
      · left; exact ⟨n, hn₁, hnpow⟩
      · right; exact ⟨n, hn₂, hnpow⟩
    · rintro x (⟨n, hn⟩ | ⟨n, hn⟩)
      · exact ⟨n, Or.inl hn.1, hn.2⟩
      · exact ⟨n, Or.inr hn.1, hn.2⟩
  /-
    S T : Set PNat
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    h : IsCyclotomicExtension (Union.union S T) A B
    this : Eq (setOf fun b => Exists fun n => And (Membership.mem (Union.union S T …
    ⊢ IsCyclotomicExtension T (Subtype fun x => Membership.mem (Algebra.adjoin A ( …
  -/
  refine ⟨fun hn => ((isCyclotomicExtension_iff _ A _).1 h).1 (mem_union_right S hn), fun b => ?_⟩
  /-
    S T : Set PNat
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    h : IsCyclotomicExtension (Union.union S T) A B
    this : Eq (setOf fun b => Exists fun n => And (Membership.mem (Union.union S T …
    b : B
    ⊢ Membership.mem (Algebra.adjoin (Subtype fun x => Membership.mem (Algebra.adj …
  -/
  replace h := ((isCyclotomicExtension_iff _ _ _).1 h).2 b
  /-
    S T : Set PNat
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    this : Eq (setOf fun b => Exists fun n => And (Membership.mem (Union.union S T …
    b : B
    h : Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n => And (Memb …
    ⊢ Membership.mem (Algebra.adjoin (Subtype fun x => Membership.mem (Algebra.adj …
  -/
  rwa [this, adjoin_union_eq_adjoin_adjoin, Subalgebra.mem_restrictScalars] at h
  /-
    🎉 no goals
  -/


/-- If `B` is a cyclotomic extension of `A` given by roots of unity of order in `T` and `S ⊆ T`,
then `adjoin A { b : B | ∃ a : ℕ+, a ∈ S ∧ b ^ (a : ℕ) = 1 }` is a cyclotomic extension of `B`
given by roots of unity of order in `S`. -/
theorem union_left [h : IsCyclotomicExtension T A B] (hS : S ⊆ T) :
    IsCyclotomicExtension S A (adjoin A {b : B | ∃ a : ℕ+, a ∈ S ∧ b ^ (a : ℕ) = 1}) := by
  /-
    S T : Set PNat
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    h : IsCyclotomicExtension T A B
    hS : HasSubset.Subset S T
    ⊢ IsCyclotomicExtension S A (Subtype fun x => Membership.mem (Algebra.adjoin A …
  -/
  refine ⟨@fun n hn => ?_, fun b => ?_⟩
    /-
      case refine_1
      S T : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : IsCyclotomicExtension T A B
      hS : HasSubset.Subset S T
      n : PNat
      hn : Membership.mem S n
      ⊢ Exists fun r => IsPrimitiveRoot r ↑n
    -/
  · obtain ⟨b, hb⟩ := ((isCyclotomicExtension_iff _ _ _).1 h).1 (hS hn)
    /-
      case refine_1.intro
      S T : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : IsCyclotomicExtension T A B
      hS : HasSubset.Subset S T
      n : PNat
      hn : Membership.mem S n
      b : B
      hb : IsPrimitiveRoot b ↑n
      ⊢ Exists fun r => IsPrimitiveRoot r ↑n
    -/
    refine ⟨⟨b, subset_adjoin ⟨n, hn, hb.pow_eq_one⟩⟩, ?_⟩
    /-
      case refine_1.intro
      S T : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : IsCyclotomicExtension T A B
      hS : HasSubset.Subset S T
      n : PNat
      hn : Membership.mem S n
      b : B
      hb : IsPrimitiveRoot b ↑n
      ⊢ IsPrimitiveRoot ⟨b, ⋯⟩ ↑n
    -/
    rwa [← IsPrimitiveRoot.coe_submonoidClass_iff, Subtype.coe_mk]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      S T : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : IsCyclotomicExtension T A B
      hS : HasSubset.Subset S T
      b : Subtype fun x => Membership.mem (Algebra.adjoin A (setOf fun b => Exists f …
      ⊢ Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n => And (Member …
    -/
  · convert mem_top (R := A) (x := b)
    /-
      case h.e'_4
      S T : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : IsCyclotomicExtension T A B
      hS : HasSubset.Subset S T
      b : Subtype fun x => Membership.mem (Algebra.adjoin A (setOf fun b => Exists f …
      ⊢ Eq (Algebra.adjoin A (setOf fun b => Exists fun n => And (Membership.mem S n …
    -/
    rw [← adjoin_adjoin_coe_preimage, preimage_setOf_eq]
    /-
      case h.e'_4
      S T : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : IsCyclotomicExtension T A B
      hS : HasSubset.Subset S T
      b : Subtype fun x => Membership.mem (Algebra.adjoin A (setOf fun b => Exists f …
      ⊢ Eq (Algebra.adjoin A (setOf fun b => Exists fun n => And (Membership.mem S n …
    -/
    norm_cast
    /-
      🎉 no goals
    -/


/-- If `∀ s ∈ S, n ∣ s` and `S` is not empty, then `IsCyclotomicExtension S A B` implies
`IsCyclotomicExtension (S ∪ {n}) A B`. -/
theorem of_union_of_dvd (h : ∀ s ∈ S, n ∣ s) (hS : S.Nonempty) [H : IsCyclotomicExtension S A B] :
    IsCyclotomicExtension (S ∪ {n}) A B := by
  /-
    n : PNat
    S : Set PNat
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd n s
    hS : S.Nonempty
    H : IsCyclotomicExtension S A B
    ⊢ IsCyclotomicExtension (Union.union S (Singleton.singleton n)) A B
  -/
  refine (iff_adjoin_eq_top _ A _).2 ⟨fun s hs => ?_, ?_⟩
    /-
      case refine_1
      n : PNat
      S : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd n s
      hS : S.Nonempty
      H : IsCyclotomicExtension S A B
      s : PNat
      hs : Membership.mem (Union.union S (Singleton.singleton n)) s
      ⊢ Exists fun r => IsPrimitiveRoot r ↑s
    -/
  · rw [mem_union, mem_singleton_iff] at hs
    /-
      case refine_1
      n : PNat
      S : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd n s
      hS : S.Nonempty
      H : IsCyclotomicExtension S A B
      s : PNat
      hs : Or (Membership.mem S s) (Eq s n)
      ⊢ Exists fun r => IsPrimitiveRoot r ↑s
    -/
    obtain hs | rfl := hs
      /-
        case refine_1.inl
        n : PNat
        S : Set PNat
        A : Type u
        B : Type v
        inst✝² : CommRing A
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd n s
        hS : S.Nonempty
        H : IsCyclotomicExtension S A B
        s : PNat
        hs : Membership.mem S s
        ⊢ Exists fun r => IsPrimitiveRoot r ↑s
      -/
    · exact H.exists_prim_root hs
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr
        S : Set PNat
        A : Type u
        B : Type v
        inst✝² : CommRing A
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        hS : S.Nonempty
        H : IsCyclotomicExtension S A B
        s : PNat
        h : ∀ (s_1 : PNat), Membership.mem S s_1 → Dvd.dvd s s_1
        ⊢ Exists fun r => IsPrimitiveRoot r ↑s
      -/
    · obtain ⟨m, hm⟩ := hS
      /-
        case refine_1.inr.intro
        S : Set PNat
        A : Type u
        B : Type v
        inst✝² : CommRing A
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        H : IsCyclotomicExtension S A B
        s : PNat
        h : ∀ (s_1 : PNat), Membership.mem S s_1 → Dvd.dvd s s_1
        m : PNat
        hm : Membership.mem S m
        ⊢ Exists fun r => IsPrimitiveRoot r ↑s
      -/
      obtain ⟨x, rfl⟩ := h m hm
      /-
        case refine_1.inr.intro.intro
        S : Set PNat
        A : Type u
        B : Type v
        inst✝² : CommRing A
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        H : IsCyclotomicExtension S A B
        s : PNat
        h : ∀ (s_1 : PNat), Membership.mem S s_1 → Dvd.dvd s s_1
        x : PNat
        hm : Membership.mem S (HMul.hMul s x)
        ⊢ Exists fun r => IsPrimitiveRoot r ↑s
      -/
      obtain ⟨ζ, hζ⟩ := H.exists_prim_root hm
      /-
        case refine_1.inr.intro.intro.intro
        S : Set PNat
        A : Type u
        B : Type v
        inst✝² : CommRing A
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        H : IsCyclotomicExtension S A B
        s : PNat
        h : ∀ (s_1 : PNat), Membership.mem S s_1 → Dvd.dvd s s_1
        x : PNat
        hm : Membership.mem S (HMul.hMul s x)
        ζ : B
        hζ : IsPrimitiveRoot ζ ↑(HMul.hMul s x)
        ⊢ Exists fun r => IsPrimitiveRoot r ↑s
      -/
      refine ⟨ζ ^ (x : ℕ), ?_⟩
      /-
        case refine_1.inr.intro.intro.intro
        S : Set PNat
        A : Type u
        B : Type v
        inst✝² : CommRing A
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        H : IsCyclotomicExtension S A B
        s : PNat
        h : ∀ (s_1 : PNat), Membership.mem S s_1 → Dvd.dvd s s_1
        x : PNat
        hm : Membership.mem S (HMul.hMul s x)
        ζ : B
        hζ : IsPrimitiveRoot ζ ↑(HMul.hMul s x)
        ⊢ IsPrimitiveRoot (HPow.hPow ζ ↑x) ↑s
      -/
      convert hζ.pow_of_dvd x.ne_zero (dvd_mul_left (x : ℕ) s)
      /-
        case h.e'_4
        S : Set PNat
        A : Type u
        B : Type v
        inst✝² : CommRing A
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        H : IsCyclotomicExtension S A B
        s : PNat
        h : ∀ (s_1 : PNat), Membership.mem S s_1 → Dvd.dvd s s_1
        x : PNat
        hm : Membership.mem S (HMul.hMul s x)
        ζ : B
        hζ : IsPrimitiveRoot ζ ↑(HMul.hMul s x)
        ⊢ Eq (↑s) (HDiv.hDiv ↑(HMul.hMul s x) ↑x)
      -/
      simp only [PNat.mul_coe, Nat.mul_div_left, PNat.pos]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      n : PNat
      S : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd n s
      hS : S.Nonempty
      H : IsCyclotomicExtension S A B
      ⊢ Eq (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (Membership.mem ( …
    -/
  · refine _root_.eq_top_iff.2 ?_
    /-
      case refine_2
      n : PNat
      S : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd n s
      hS : S.Nonempty
      H : IsCyclotomicExtension S A B
      ⊢ LE.le Top.top (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (Membe …
    -/
    rw [← ((iff_adjoin_eq_top S A B).1 H).2]
    /-
      case refine_2
      n : PNat
      S : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd n s
      hS : S.Nonempty
      H : IsCyclotomicExtension S A B
      ⊢ LE.le (Algebra.adjoin A (setOf fun b => Exists fun n => And (Membership.mem  …
    -/
    refine adjoin_mono fun x hx => ?_
    /-
      case refine_2
      n : PNat
      S : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd n s
      hS : S.Nonempty
      H : IsCyclotomicExtension S A B
      x : B
      hx : Membership.mem (setOf fun b => Exists fun n => And (Membership.mem S n) ( …
      ⊢ Membership.mem (setOf fun b => Exists fun n_1 => And (Membership.mem (Union. …
    -/
    simp only [union_singleton, mem_insert_iff, mem_setOf_eq] at hx ⊢
    /-
      case refine_2
      n : PNat
      S : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd n s
      hS : S.Nonempty
      H : IsCyclotomicExtension S A B
      x : B
      hx : Exists fun n => And (Membership.mem S n) (Eq (HPow.hPow x ↑n) 1)
      ⊢ Exists fun n_1 => And (Or (Eq n_1 n) (Membership.mem S n_1)) (Eq (HPow.hPow  …
    -/
    obtain ⟨m, hm⟩ := hx
    /-
      case refine_2.intro
      n : PNat
      S : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd n s
      hS : S.Nonempty
      H : IsCyclotomicExtension S A B
      x : B
      m : PNat
      hm : And (Membership.mem S m) (Eq (HPow.hPow x ↑m) 1)
      ⊢ Exists fun n_1 => And (Or (Eq n_1 n) (Membership.mem S n_1)) (Eq (HPow.hPow  …
    -/
    exact ⟨m, ⟨Or.inr hm.1, hm.2⟩⟩
    /-
      🎉 no goals
    -/


/-- If `∀ s ∈ S, n ∣ s` and `S` is not empty, then `IsCyclotomicExtension S A B` if and only if
`IsCyclotomicExtension (S ∪ {n}) A B`. -/
theorem iff_union_of_dvd (h : ∀ s ∈ S, n ∣ s) (hS : S.Nonempty) :
    IsCyclotomicExtension S A B ↔ IsCyclotomicExtension (S ∪ {n}) A B := by
  refine
    ⟨fun H => of_union_of_dvd A B h hS, fun H => (iff_adjoin_eq_top _ A _).2 ⟨fun s hs => ?_, ?_⟩⟩
    /-
      case refine_1
      n : PNat
      S : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd n s
      hS : S.Nonempty
      H : IsCyclotomicExtension (Union.union S (Singleton.singleton n)) A B
      s : PNat
      hs : Membership.mem S s
      ⊢ Exists fun r => IsPrimitiveRoot r ↑s
    -/
  · exact H.exists_prim_root (subset_union_left hs)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : PNat
      S : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd n s
      hS : S.Nonempty
      H : IsCyclotomicExtension (Union.union S (Singleton.singleton n)) A B
      ⊢ Eq (Algebra.adjoin A (setOf fun b => Exists fun n => And (Membership.mem S n …
    -/
  · rw [_root_.eq_top_iff, ← ((iff_adjoin_eq_top _ A B).1 H).2]
    /-
      case refine_2
      n : PNat
      S : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd n s
      hS : S.Nonempty
      H : IsCyclotomicExtension (Union.union S (Singleton.singleton n)) A B
      ⊢ LE.le (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (Membership.me …
    -/
    refine adjoin_mono fun x hx => ?_
    /-
      case refine_2
      n : PNat
      S : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd n s
      hS : S.Nonempty
      H : IsCyclotomicExtension (Union.union S (Singleton.singleton n)) A B
      x : B
      hx : Membership.mem (setOf fun b => Exists fun n_1 => And (Membership.mem (Uni …
      ⊢ Membership.mem (setOf fun b => Exists fun n => And (Membership.mem S n) (Eq  …
    -/
    simp only [union_singleton, mem_insert_iff, mem_setOf_eq] at hx ⊢
    /-
      case refine_2
      n : PNat
      S : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd n s
      hS : S.Nonempty
      H : IsCyclotomicExtension (Union.union S (Singleton.singleton n)) A B
      x : B
      hx : Exists fun n_1 => And (Or (Eq n_1 n) (Membership.mem S n_1)) (Eq (HPow.hP …
      ⊢ Exists fun n => And (Membership.mem S n) (Eq (HPow.hPow x ↑n) 1)
    -/
    obtain ⟨m, rfl | hm, hxpow⟩ := hx
      /-
        case refine_2.intro.intro.inl
        S : Set PNat
        A : Type u
        B : Type v
        inst✝² : CommRing A
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        hS : S.Nonempty
        x : B
        m : PNat
        hxpow : Eq (HPow.hPow x ↑m) 1
        h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd m s
        H : IsCyclotomicExtension (Union.union S (Singleton.singleton m)) A B
        ⊢ Exists fun n => And (Membership.mem S n) (Eq (HPow.hPow x ↑n) 1)
      -/
    · obtain ⟨y, hy⟩ := hS
      /-
        case refine_2.intro.intro.inl.intro
        S : Set PNat
        A : Type u
        B : Type v
        inst✝² : CommRing A
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        x : B
        m : PNat
        hxpow : Eq (HPow.hPow x ↑m) 1
        h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd m s
        H : IsCyclotomicExtension (Union.union S (Singleton.singleton m)) A B
        y : PNat
        hy : Membership.mem S y
        ⊢ Exists fun n => And (Membership.mem S n) (Eq (HPow.hPow x ↑n) 1)
      -/
      refine ⟨y, ⟨hy, ?_⟩⟩
      /-
        case refine_2.intro.intro.inl.intro
        S : Set PNat
        A : Type u
        B : Type v
        inst✝² : CommRing A
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        x : B
        m : PNat
        hxpow : Eq (HPow.hPow x ↑m) 1
        h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd m s
        H : IsCyclotomicExtension (Union.union S (Singleton.singleton m)) A B
        y : PNat
        hy : Membership.mem S y
        ⊢ Eq (HPow.hPow x ↑y) 1
      -/
      obtain ⟨z, rfl⟩ := h y hy
      /-
        case refine_2.intro.intro.inl.intro.intro
        S : Set PNat
        A : Type u
        B : Type v
        inst✝² : CommRing A
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        x : B
        m : PNat
        hxpow : Eq (HPow.hPow x ↑m) 1
        h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd m s
        H : IsCyclotomicExtension (Union.union S (Singleton.singleton m)) A B
        z : PNat
        hy : Membership.mem S (HMul.hMul m z)
        ⊢ Eq (HPow.hPow x ↑(HMul.hMul m z)) 1
      -/
      simp only [PNat.mul_coe, pow_mul, hxpow, one_pow]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.inr
        n : PNat
        S : Set PNat
        A : Type u
        B : Type v
        inst✝² : CommRing A
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        h : ∀ (s : PNat), Membership.mem S s → Dvd.dvd n s
        hS : S.Nonempty
        H : IsCyclotomicExtension (Union.union S (Singleton.singleton n)) A B
        x : B
        m : PNat
        hxpow : Eq (HPow.hPow x ↑m) 1
        hm : Membership.mem S m
        ⊢ Exists fun n => And (Membership.mem S n) (Eq (HPow.hPow x ↑n) 1)
      -/
    · exact ⟨m, ⟨hm, hxpow⟩⟩
      /-
        🎉 no goals
      -/


/-- `IsCyclotomicExtension S A B` is equivalent to `IsCyclotomicExtension (S ∪ {1}) A B`. -/
theorem iff_union_singleton_one :
    IsCyclotomicExtension S A B ↔ IsCyclotomicExtension (S ∪ {1}) A B := by
  /-
    S : Set PNat
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    ⊢ Iff (IsCyclotomicExtension S A B) (IsCyclotomicExtension (Union.union S (Sin …
  -/
  obtain hS | rfl := S.eq_empty_or_nonempty.symm
    /-
      case inl
      S : Set PNat
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      hS : S.Nonempty
      ⊢ Iff (IsCyclotomicExtension S A B) (IsCyclotomicExtension (Union.union S (Sin …
    -/
  · exact iff_union_of_dvd _ _ (fun s _ => one_dvd _) hS
    /-
      🎉 no goals
    -/
  /-
    case inr
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    ⊢ Iff (IsCyclotomicExtension EmptyCollection.emptyCollection A B) (IsCyclotomi …
  -/
  rw [empty_union]
  /-
    case inr
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    ⊢ Iff (IsCyclotomicExtension EmptyCollection.emptyCollection A B) (IsCyclotomi …
  -/
  refine ⟨fun H => ?_, fun H => ?_⟩
    /-
      case inr.refine_1
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      H : IsCyclotomicExtension EmptyCollection.emptyCollection A B
      ⊢ IsCyclotomicExtension (Singleton.singleton 1) A B
    -/
  · refine (iff_adjoin_eq_top _ A _).2 ⟨fun s hs => ⟨1, by simp [mem_singleton_iff.1 hs]⟩, ?_⟩
    /-
      case inr.refine_1
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      H : IsCyclotomicExtension EmptyCollection.emptyCollection A B
      ⊢ Eq (Algebra.adjoin A (setOf fun b => Exists fun n => And (Membership.mem (Si …
    -/
    simp [adjoin_singleton_one, empty]
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_2
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      H : IsCyclotomicExtension (Singleton.singleton 1) A B
      ⊢ IsCyclotomicExtension EmptyCollection.emptyCollection A B
    -/
  · refine (iff_adjoin_eq_top _ A _).2 ⟨fun s hs => (not_mem_empty s hs).elim, ?_⟩
    /-
      case inr.refine_2
      A : Type u
      B : Type v
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      H : IsCyclotomicExtension (Singleton.singleton 1) A B
      ⊢ Eq (Algebra.adjoin A (setOf fun b => Exists fun n => And (Membership.mem Emp …
    -/
    simp [@singleton_one A B _ _ _ H]
    /-
      🎉 no goals
    -/


/-- If `(⊥ : SubAlgebra A B) = ⊤`, then `IsCyclotomicExtension {1} A B`. -/
theorem singleton_one_of_bot_eq_top (h : (⊥ : Subalgebra A B) = ⊤) :
    IsCyclotomicExtension {1} A B := by
  /-
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    h : Eq Bot.bot Top.top
    ⊢ IsCyclotomicExtension (Singleton.singleton 1) A B
  -/
  convert (iff_union_singleton_one _ A _).1 (singleton_zero_of_bot_eq_top h)
  /-
    case h.e'_1
    A : Type u
    B : Type v
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    h : Eq Bot.bot Top.top
    ⊢ Eq (Singleton.singleton 1) (Union.union EmptyCollection.emptyCollection (Sin …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If `Function.Surjective (algebraMap A B)`, then `IsCyclotomicExtension {1} A B`. -/
theorem singleton_one_of_algebraMap_bijective (h : Function.Surjective (algebraMap A B)) :
    IsCyclotomicExtension {1} A B :=
  singleton_one_of_bot_eq_top (surjective_algebraMap_iff.1 h).symm


/-- Given `(f : B ≃ₐ[A] C)`, if `IsCyclotomicExtension S A B` then
`IsCyclotomicExtension S A C`. -/
protected
theorem equiv {C : Type*} [CommRing C] [Algebra A C] [h : IsCyclotomicExtension S A B]
    (f : B ≃ₐ[A] C) : IsCyclotomicExtension S A C := by
  /-
    S : Set PNat
    A : Type u
    B : Type v
    inst✝⁴ : CommRing A
    inst✝³ : CommRing B
    inst✝² : Algebra A B
    C : Type u_1
    inst✝¹ : CommRing C
    inst✝ : Algebra A C
    h : IsCyclotomicExtension S A B
    f : AlgEquiv A B C
    ⊢ IsCyclotomicExtension S A C
  -/
  letI : Algebra B C := f.toAlgHom.toRingHom.toAlgebra
  /-
    S : Set PNat
    A : Type u
    B : Type v
    inst✝⁴ : CommRing A
    inst✝³ : CommRing B
    inst✝² : Algebra A B
    C : Type u_1
    inst✝¹ : CommRing C
    inst✝ : Algebra A C
    h : IsCyclotomicExtension S A B
    f : AlgEquiv A B C
    this : Algebra B C := (↑f).toAlgebra
    ⊢ IsCyclotomicExtension S A C
  -/
  haveI : IsCyclotomicExtension {1} B C := singleton_one_of_algebraMap_bijective f.surjective
  /-
    S : Set PNat
    A : Type u
    B : Type v
    inst✝⁴ : CommRing A
    inst✝³ : CommRing B
    inst✝² : Algebra A B
    C : Type u_1
    inst✝¹ : CommRing C
    inst✝ : Algebra A C
    h : IsCyclotomicExtension S A B
    f : AlgEquiv A B C
    this✝ : Algebra B C := (↑f).toAlgebra
    this : IsCyclotomicExtension (Singleton.singleton 1) B C
    ⊢ IsCyclotomicExtension S A C
  -/
  haveI : IsScalarTower A B C := IsScalarTower.of_algHom f.toAlgHom
  /-
    S : Set PNat
    A : Type u
    B : Type v
    inst✝⁴ : CommRing A
    inst✝³ : CommRing B
    inst✝² : Algebra A B
    C : Type u_1
    inst✝¹ : CommRing C
    inst✝ : Algebra A C
    h : IsCyclotomicExtension S A B
    f : AlgEquiv A B C
    this✝¹ : Algebra B C := (↑f).toAlgebra
    this✝ : IsCyclotomicExtension (Singleton.singleton 1) B C
    this : IsScalarTower A B C
    ⊢ IsCyclotomicExtension S A C
  -/
  exact (iff_union_singleton_one _ _ _).2 (trans S {1} A B C f.injective)
  /-
    🎉 no goals
  -/


protected
theorem neZero [h : IsCyclotomicExtension {n} A B] [IsDomain B] : NeZero ((n : ℕ) : B) := by
  /-
    n : PNat
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    h : IsCyclotomicExtension (Singleton.singleton n) A B
    inst✝ : IsDomain B
    ⊢ NeZero ↑↑n
  -/
  obtain ⟨⟨r, hr⟩, -⟩ := (iff_singleton n A B).1 h
  /-
    case intro.intro
    n : PNat
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    h : IsCyclotomicExtension (Singleton.singleton n) A B
    inst✝ : IsDomain B
    r : B
    hr : IsPrimitiveRoot r ↑n
    ⊢ NeZero ↑↑n
  -/
  exact hr.neZero'
  /-
    🎉 no goals
  -/


protected
theorem neZero' [IsCyclotomicExtension {n} A B] [IsDomain B] : NeZero ((n : ℕ) : A) := by
  /-
    n : PNat
    A : Type u
    B : Type v
    inst✝⁴ : CommRing A
    inst✝³ : CommRing B
    inst✝² : Algebra A B
    inst✝¹ : IsCyclotomicExtension (Singleton.singleton n) A B
    inst✝ : IsDomain B
    ⊢ NeZero ↑↑n
  -/
  haveI := IsCyclotomicExtension.neZero n A B
  /-
    n : PNat
    A : Type u
    B : Type v
    inst✝⁴ : CommRing A
    inst✝³ : CommRing B
    inst✝² : Algebra A B
    inst✝¹ : IsCyclotomicExtension (Singleton.singleton n) A B
    inst✝ : IsDomain B
    this : NeZero ↑↑n
    ⊢ NeZero ↑↑n
  -/
  exact NeZero.nat_of_neZero (algebraMap A B)
  /-
    🎉 no goals
  -/


theorem finite_of_singleton [IsDomain B] [h : IsCyclotomicExtension {n} A B] :
    Module.Finite A B := by
  classical
  rw [Module.finite_def, ← top_toSubmodule, ← ((iff_adjoin_eq_top _ _ _).1 h).2]
  refine fg_adjoin_of_finite ?_ fun b hb => ?_
  · simp only [mem_singleton_iff, exists_eq_left]
    have : {b : B | b ^ (n : ℕ) = 1} = (nthRoots n (1 : B)).toFinset :=
      Set.ext fun x => ⟨fun h => by simpa using h, fun h => by simpa using h⟩
    rw [this]
    exact (nthRoots (↑n) 1).toFinset.finite_toSet
  · simp only [mem_singleton_iff, exists_eq_left, mem_setOf_eq] at hb
    exact ⟨X ^ (n : ℕ) - 1, ⟨monic_X_pow_sub_C _ n.pos.ne.symm, by simp [hb]⟩⟩


/-- If `S` is finite and `IsCyclotomicExtension S A B`, then `B` is a finite `A`-algebra. -/
protected theorem finite [IsDomain B] [h₁ : Finite S] [h₂ : IsCyclotomicExtension S A B] :
    Module.Finite A B := by
  /-
    S : Set PNat
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : IsDomain B
    h₁ : Finite ↑S
    h₂ : IsCyclotomicExtension S A B
    ⊢ Module.Finite A B
  -/
  cases' nonempty_fintype S with h
  /-
    case intro
    S : Set PNat
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : IsDomain B
    h₁ : Finite ↑S
    h₂ : IsCyclotomicExtension S A B
    h : Fintype ↑S
    ⊢ Module.Finite A B
  -/
  revert h₂ A B
  /-
    case intro
    S : Set PNat
    h₁ : Finite ↑S
    h : Fintype ↑S
    ⊢ ∀ (A : Type u) (B : Type v) [inst : CommRing A] [inst_1 : CommRing B] [inst_ …
  -/
  refine Set.Finite.induction_on h₁ (fun A B => ?_) @fun n S _ _ H A B => ?_
    /-
      case intro.refine_1
      S : Set PNat
      h₁ : Finite ↑S
      h : Fintype ↑S
      A : Type u
      B : Type v
      ⊢ ∀ [inst : CommRing A] [inst_1 : CommRing B] [inst_2 : Algebra A B] [inst_3 : …
    -/
  · intro _ _ _ _ _
    /-
      case intro.refine_1
      S : Set PNat
      h₁ : Finite ↑S
      h : Fintype ↑S
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsDomain B
      h₂✝ : IsCyclotomicExtension EmptyCollection.emptyCollection A B
      ⊢ Module.Finite A B
    -/
    refine Module.finite_def.2 ⟨({1} : Finset B), ?_⟩
    /-
      case intro.refine_1
      S : Set PNat
      h₁ : Finite ↑S
      h : Fintype ↑S
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsDomain B
      h₂✝ : IsCyclotomicExtension EmptyCollection.emptyCollection A B
      ⊢ Eq (Submodule.span A ↑(Singleton.singleton 1)) Top.top
    -/
    simp [← top_toSubmodule, ← empty, toSubmodule_bot, Submodule.one_eq_span]
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      S✝ : Set PNat
      h₁ : Finite ↑S✝
      h : Fintype ↑S✝
      n : PNat
      S : Set PNat
      x✝¹ : Not (Membership.mem S n)
      x✝ : S.Finite
      H : ∀ (A : Type u) (B : Type v) [inst : CommRing A] [inst_1 : CommRing B] [ins …
      A : Type u
      B : Type v
      ⊢ ∀ [inst : CommRing A] [inst_1 : CommRing B] [inst_2 : Algebra A B] [inst_3 : …
    -/
  · intro _ _ _ _ h
    haveI : IsCyclotomicExtension S A (adjoin A {b : B | ∃ n : ℕ+, n ∈ S ∧ b ^ (n : ℕ) = 1}) :=
      union_left _ (insert n S) _ _ (subset_insert n S)
    /-
      case intro.refine_2
      S✝ : Set PNat
      h₁ : Finite ↑S✝
      h✝ : Fintype ↑S✝
      n : PNat
      S : Set PNat
      x✝¹ : Not (Membership.mem S n)
      x✝ : S.Finite
      H : ∀ (A : Type u) (B : Type v) [inst : CommRing A] [inst_1 : CommRing B] [ins …
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsDomain B
      h : IsCyclotomicExtension (Insert.insert n S) A B
      this : IsCyclotomicExtension S A (Subtype fun x => Membership.mem (Algebra.adj …
      ⊢ Module.Finite A B
    -/
    haveI := H A (adjoin A {b : B | ∃ n : ℕ+, n ∈ S ∧ b ^ (n : ℕ) = 1})
    have : Module.Finite (adjoin A {b : B | ∃ n : ℕ+, n ∈ S ∧ b ^ (n : ℕ) = 1}) B := by
      rw [← union_singleton] at h
      letI := @union_right S {n} A B _ _ _ h
      exact finite_of_singleton n _ _
    /-
      case intro.refine_2
      S✝ : Set PNat
      h₁ : Finite ↑S✝
      h✝ : Fintype ↑S✝
      n : PNat
      S : Set PNat
      x✝¹ : Not (Membership.mem S n)
      x✝ : S.Finite
      H : ∀ (A : Type u) (B : Type v) [inst : CommRing A] [inst_1 : CommRing B] [ins …
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsDomain B
      h : IsCyclotomicExtension (Insert.insert n S) A B
      this✝¹ : IsCyclotomicExtension S A (Subtype fun x => Membership.mem (Algebra.a …
      this✝ : Module.Finite A (Subtype fun x => Membership.mem (Algebra.adjoin A (se …
      this : Module.Finite (Subtype fun x => Membership.mem (Algebra.adjoin A (setOf …
      ⊢ Module.Finite A B
    -/
    exact Module.Finite.trans (adjoin A {b : B | ∃ n : ℕ+, n ∈ S ∧ b ^ (n : ℕ) = 1}) _
    /-
      🎉 no goals
    -/


/-- A cyclotomic finite extension of a number field is a number field. -/
theorem numberField [h : NumberField K] [Finite S] [IsCyclotomicExtension S K L] : NumberField L :=
  { to_charZero := charZero_of_injective_algebraMap (algebraMap K L).injective
    to_finiteDimensional := by
      /-
        S : Set PNat
        K : Type w
        L : Type z
        inst✝⁴ : Field K
        inst✝³ : Field L
        inst✝² : Algebra K L
        h : NumberField K
        inst✝¹ : Finite ↑S
        inst✝ : IsCyclotomicExtension S K L
        ⊢ FiniteDimensional Rat L
      -/
      haveI := charZero_of_injective_algebraMap (algebraMap K L).injective
      /-
        S : Set PNat
        K : Type w
        L : Type z
        inst✝⁴ : Field K
        inst✝³ : Field L
        inst✝² : Algebra K L
        h : NumberField K
        inst✝¹ : Finite ↑S
        inst✝ : IsCyclotomicExtension S K L
        this : CharZero L
        ⊢ FiniteDimensional Rat L
      -/
      haveI := IsCyclotomicExtension.finite S K L
      /-
        S : Set PNat
        K : Type w
        L : Type z
        inst✝⁴ : Field K
        inst✝³ : Field L
        inst✝² : Algebra K L
        h : NumberField K
        inst✝¹ : Finite ↑S
        inst✝ : IsCyclotomicExtension S K L
        this✝ : CharZero L
        this : Module.Finite K L
        ⊢ FiniteDimensional Rat L
      -/
      exact Module.Finite.trans K _ }
      /-
        🎉 no goals
      -/


/-- A finite cyclotomic extension of an integral noetherian domain is integral -/
theorem integral [IsDomain B] [IsNoetherianRing A] [Finite S] [IsCyclotomicExtension S A B] :
    Algebra.IsIntegral A B :=
  have := IsCyclotomicExtension.finite S A B
  ⟨isIntegral_of_noetherian inferInstance⟩


/-- If `S` is finite and `IsCyclotomicExtension S K A`, then `finiteDimensional K A`. -/
theorem finiteDimensional (C : Type z) [Finite S] [CommRing C] [Algebra K C] [IsDomain C]
    [IsCyclotomicExtension S K C] : FiniteDimensional K C :=
  IsCyclotomicExtension.finite S K C


theorem adjoin_roots_cyclotomic_eq_adjoin_nth_roots [IsDomain B] {ζ : B} {n : ℕ+}
    (hζ : IsPrimitiveRoot ζ n) :
    adjoin A ((cyclotomic n A).rootSet B) =
      adjoin A {b : B | ∃ a : ℕ+, a ∈ ({n} : Set ℕ+) ∧ b ^ (a : ℕ) = 1} := by
  /-
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : IsDomain B
    ζ : B
    n : PNat
    hζ : IsPrimitiveRoot ζ ↑n
    ⊢ Eq (Algebra.adjoin A ((Polynomial.cyclotomic (↑n) A).rootSet B)) (Algebra.ad …
  -/
  simp only [mem_singleton_iff, exists_eq_left, map_cyclotomic]
  /-
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : IsDomain B
    ζ : B
    n : PNat
    hζ : IsPrimitiveRoot ζ ↑n
    ⊢ Eq (Algebra.adjoin A ((Polynomial.cyclotomic (↑n) A).rootSet B)) (Algebra.ad …
  -/
  refine le_antisymm (adjoin_mono fun x hx => ?_) (adjoin_le fun x hx => ?_)
    /-
      case refine_1
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsDomain B
      ζ : B
      n : PNat
      hζ : IsPrimitiveRoot ζ ↑n
      x : B
      hx : Membership.mem ((Polynomial.cyclotomic (↑n) A).rootSet B) x
      ⊢ Membership.mem (setOf fun b => Eq (HPow.hPow b ↑n) 1) x
    -/
  · rw [mem_rootSet'] at hx
    /-
      case refine_1
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsDomain B
      ζ : B
      n : PNat
      hζ : IsPrimitiveRoot ζ ↑n
      x : B
      hx : And (Ne (Polynomial.map (algebraMap A B) (Polynomial.cyclotomic (↑n) A))  …
      ⊢ Membership.mem (setOf fun b => Eq (HPow.hPow b ↑n) 1) x
    -/
    simp only [mem_singleton_iff, exists_eq_left, mem_setOf_eq]
    /-
      case refine_1
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsDomain B
      ζ : B
      n : PNat
      hζ : IsPrimitiveRoot ζ ↑n
      x : B
      hx : And (Ne (Polynomial.map (algebraMap A B) (Polynomial.cyclotomic (↑n) A))  …
      ⊢ Eq (HPow.hPow x ↑n) 1
    -/
    rw [isRoot_of_unity_iff n.pos]
    /-
      case refine_1
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsDomain B
      ζ : B
      n : PNat
      hζ : IsPrimitiveRoot ζ ↑n
      x : B
      hx : And (Ne (Polynomial.map (algebraMap A B) (Polynomial.cyclotomic (↑n) A))  …
      ⊢ Exists fun i => And (Membership.mem (↑n).divisors i) ((Polynomial.cyclotomic …
    -/
    refine ⟨n, Nat.mem_divisors_self n n.ne_zero, ?_⟩
    /-
      case refine_1
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsDomain B
      ζ : B
      n : PNat
      hζ : IsPrimitiveRoot ζ ↑n
      x : B
      hx : And (Ne (Polynomial.map (algebraMap A B) (Polynomial.cyclotomic (↑n) A))  …
      ⊢ (Polynomial.cyclotomic (↑n) B).IsRoot x
    -/
    rw [IsRoot.def, ← map_cyclotomic n (algebraMap A B), eval_map, ← aeval_def]
    /-
      case refine_1
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsDomain B
      ζ : B
      n : PNat
      hζ : IsPrimitiveRoot ζ ↑n
      x : B
      hx : And (Ne (Polynomial.map (algebraMap A B) (Polynomial.cyclotomic (↑n) A))  …
      ⊢ Eq ((Polynomial.aeval x) (Polynomial.cyclotomic (↑n) A)) 0
    -/
    exact hx.2
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsDomain B
      ζ : B
      n : PNat
      hζ : IsPrimitiveRoot ζ ↑n
      x : B
      hx : Membership.mem (setOf fun b => Eq (HPow.hPow b ↑n) 1) x
      ⊢ Membership.mem (↑(Algebra.adjoin A ((Polynomial.cyclotomic (↑n) A).rootSet B …
    -/
  · simp only [mem_singleton_iff, exists_eq_left, mem_setOf_eq] at hx
    /-
      case refine_2
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsDomain B
      ζ : B
      n : PNat
      hζ : IsPrimitiveRoot ζ ↑n
      x : B
      hx : Eq (HPow.hPow x ↑n) 1
      ⊢ Membership.mem (↑(Algebra.adjoin A ((Polynomial.cyclotomic (↑n) A).rootSet B …
    -/
    obtain ⟨i, _, rfl⟩ := hζ.eq_pow_of_pow_eq_one hx
    /-
      case refine_2.intro.intro
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsDomain B
      ζ : B
      n : PNat
      hζ : IsPrimitiveRoot ζ ↑n
      i : Nat
      left✝ : LT.lt i ↑n
      hx : Eq (HPow.hPow (HPow.hPow ζ i) ↑n) 1
      ⊢ Membership.mem (↑(Algebra.adjoin A ((Polynomial.cyclotomic (↑n) A).rootSet B …
    -/
    refine SetLike.mem_coe.2 (Subalgebra.pow_mem _ (subset_adjoin ?_) _)
    /-
      case refine_2.intro.intro
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsDomain B
      ζ : B
      n : PNat
      hζ : IsPrimitiveRoot ζ ↑n
      i : Nat
      left✝ : LT.lt i ↑n
      hx : Eq (HPow.hPow (HPow.hPow ζ i) ↑n) 1
      ⊢ Membership.mem ((Polynomial.cyclotomic (↑n) A).rootSet B) ζ
    -/
    rw [mem_rootSet', map_cyclotomic, aeval_def, ← eval_map, map_cyclotomic, ← IsRoot]
    /-
      case refine_2.intro.intro
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : IsDomain B
      ζ : B
      n : PNat
      hζ : IsPrimitiveRoot ζ ↑n
      i : Nat
      left✝ : LT.lt i ↑n
      hx : Eq (HPow.hPow (HPow.hPow ζ i) ↑n) 1
      ⊢ And (Ne (Polynomial.cyclotomic (↑n) B) 0) ((Polynomial.cyclotomic (↑n) B).Is …
    -/
    exact ⟨cyclotomic_ne_zero n B, hζ.isRoot_cyclotomic n.pos⟩
    /-
      🎉 no goals
    -/


theorem adjoin_roots_cyclotomic_eq_adjoin_root_cyclotomic {n : ℕ+} [IsDomain B] {ζ : B}
    (hζ : IsPrimitiveRoot ζ n) : adjoin A ((cyclotomic n A).rootSet B) = adjoin A {ζ} := by
  /-
    A : Type u
    B : Type v
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    n : PNat
    inst✝ : IsDomain B
    ζ : B
    hζ : IsPrimitiveRoot ζ ↑n
    ⊢ Eq (Algebra.adjoin A ((Polynomial.cyclotomic (↑n) A).rootSet B)) (Algebra.ad …
  -/
  refine le_antisymm (adjoin_le fun x hx => ?_) (adjoin_mono fun x hx => ?_)
  · suffices hx : x ^ n.1 = 1 by
      obtain ⟨i, _, rfl⟩ := hζ.eq_pow_of_pow_eq_one hx
      exact SetLike.mem_coe.2 (Subalgebra.pow_mem _ (subset_adjoin <| mem_singleton ζ) _)
    /-
      case refine_1
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      n : PNat
      inst✝ : IsDomain B
      ζ : B
      hζ : IsPrimitiveRoot ζ ↑n
      x : B
      hx : Membership.mem ((Polynomial.cyclotomic (↑n) A).rootSet B) x
      ⊢ Eq (HPow.hPow x ↑n) 1
    -/
    refine (isRoot_of_unity_iff n.pos B).2 ?_
    /-
      case refine_1
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      n : PNat
      inst✝ : IsDomain B
      ζ : B
      hζ : IsPrimitiveRoot ζ ↑n
      x : B
      hx : Membership.mem ((Polynomial.cyclotomic (↑n) A).rootSet B) x
      ⊢ Exists fun i => And (Membership.mem (↑n).divisors i) ((Polynomial.cyclotomic …
    -/
    refine ⟨n, Nat.mem_divisors_self n n.ne_zero, ?_⟩
    /-
      case refine_1
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      n : PNat
      inst✝ : IsDomain B
      ζ : B
      hζ : IsPrimitiveRoot ζ ↑n
      x : B
      hx : Membership.mem ((Polynomial.cyclotomic (↑n) A).rootSet B) x
      ⊢ (Polynomial.cyclotomic (↑n) B).IsRoot x
    -/
    rw [mem_rootSet', aeval_def, ← eval_map, map_cyclotomic, ← IsRoot] at hx
    /-
      case refine_1
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      n : PNat
      inst✝ : IsDomain B
      ζ : B
      hζ : IsPrimitiveRoot ζ ↑n
      x : B
      hx : And (Ne (Polynomial.cyclotomic (↑n) B) 0) ((Polynomial.cyclotomic (↑n) B) …
      ⊢ (Polynomial.cyclotomic (↑n) B).IsRoot x
    -/
    exact hx.2
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      A : Type u
      B : Type v
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      n : PNat
      inst✝ : IsDomain B
      ζ : B
      hζ : IsPrimitiveRoot ζ ↑n
      x : B
      hx : Membership.mem (Singleton.singleton ζ) x
      ⊢ Membership.mem ((Polynomial.cyclotomic (↑n) A).rootSet B) x
    -/
  · simp only [mem_singleton_iff, exists_eq_left, mem_setOf_eq] at hx
    simpa only [hx, mem_rootSet', map_cyclotomic, aeval_def, ← eval_map, IsRoot] using
      And.intro (cyclotomic_ne_zero n B) (hζ.isRoot_cyclotomic n.pos)


theorem adjoin_primitive_root_eq_top {n : ℕ+} [IsDomain B] [h : IsCyclotomicExtension {n} A B]
    {ζ : B} (hζ : IsPrimitiveRoot ζ n) : adjoin A ({ζ} : Set B) = ⊤ := by
  classical
  rw [← adjoin_roots_cyclotomic_eq_adjoin_root_cyclotomic hζ]
  rw [adjoin_roots_cyclotomic_eq_adjoin_nth_roots hζ]
  exact ((iff_adjoin_eq_top {n} A B).mp h).2


theorem _root_.IsPrimitiveRoot.adjoin_isCyclotomicExtension {ζ : B} {n : ℕ+}
    (h : IsPrimitiveRoot ζ n) : IsCyclotomicExtension {n} A (adjoin A ({ζ} : Set B)) :=
  { exists_prim_root := fun hi => by
      /-
        A : Type u
        B : Type v
        inst✝² : CommRing A
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        ζ : B
        n : PNat
        h : IsPrimitiveRoot ζ ↑n
        n✝ : PNat
        hi : Membership.mem (Singleton.singleton n) n✝
        ⊢ Exists fun r => IsPrimitiveRoot r ↑n✝
      -/
      rw [Set.mem_singleton_iff] at hi
      /-
        A : Type u
        B : Type v
        inst✝² : CommRing A
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        ζ : B
        n : PNat
        h : IsPrimitiveRoot ζ ↑n
        n✝ : PNat
        hi : Eq n✝ n
        ⊢ Exists fun r => IsPrimitiveRoot r ↑n✝
      -/
      refine ⟨⟨ζ, subset_adjoin <| Set.mem_singleton ζ⟩, ?_⟩
      /-
        A : Type u
        B : Type v
        inst✝² : CommRing A
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        ζ : B
        n : PNat
        h : IsPrimitiveRoot ζ ↑n
        n✝ : PNat
        hi : Eq n✝ n
        ⊢ IsPrimitiveRoot ⟨ζ, ⋯⟩ ↑n✝
      -/
      rwa [← IsPrimitiveRoot.coe_submonoidClass_iff, Subtype.coe_mk, hi]
      /-
        🎉 no goals
      -/
    adjoin_roots := fun ⟨x, hx⟩ => by
      refine
        adjoin_induction
          (hx := hx) (fun b hb => ?_) (fun a => ?_) (fun b₁ b₂ _ _ hb₁ hb₂ => ?_)
          (fun b₁ b₂ _ _ hb₁ hb₂ => ?_)
        /-
          case refine_1
          A : Type u
          B : Type v
          inst✝² : CommRing A
          inst✝¹ : CommRing B
          inst✝ : Algebra A B
          ζ : B
          n : PNat
          h : IsPrimitiveRoot ζ ↑n
          x✝ : Subtype fun x => Membership.mem (Algebra.adjoin A (Singleton.singleton ζ) …
          x : B
          hx : Membership.mem (Algebra.adjoin A (Singleton.singleton ζ)) x
          b : B
          hb : Membership.mem (Singleton.singleton ζ) b
          ⊢ Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (Memb …
        -/
      · rw [Set.mem_singleton_iff] at hb
        /-
          case refine_1
          A : Type u
          B : Type v
          inst✝² : CommRing A
          inst✝¹ : CommRing B
          inst✝ : Algebra A B
          ζ : B
          n : PNat
          h : IsPrimitiveRoot ζ ↑n
          x✝ : Subtype fun x => Membership.mem (Algebra.adjoin A (Singleton.singleton ζ) …
          x : B
          hx : Membership.mem (Algebra.adjoin A (Singleton.singleton ζ)) x
          b : B
          hb✝ : Membership.mem (Singleton.singleton ζ) b
          hb : Eq b ζ
          ⊢ Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (Memb …
        -/
        refine subset_adjoin ?_
        /-
          case refine_1
          A : Type u
          B : Type v
          inst✝² : CommRing A
          inst✝¹ : CommRing B
          inst✝ : Algebra A B
          ζ : B
          n : PNat
          h : IsPrimitiveRoot ζ ↑n
          x✝ : Subtype fun x => Membership.mem (Algebra.adjoin A (Singleton.singleton ζ) …
          x : B
          hx : Membership.mem (Algebra.adjoin A (Singleton.singleton ζ)) x
          b : B
          hb✝ : Membership.mem (Singleton.singleton ζ) b
          hb : Eq b ζ
          ⊢ Membership.mem (setOf fun b => Exists fun n_1 => And (Membership.mem (Single …
        -/
        simp only [mem_singleton_iff, exists_eq_left, mem_setOf_eq, hb]
        /-
          case refine_1
          A : Type u
          B : Type v
          inst✝² : CommRing A
          inst✝¹ : CommRing B
          inst✝ : Algebra A B
          ζ : B
          n : PNat
          h : IsPrimitiveRoot ζ ↑n
          x✝ : Subtype fun x => Membership.mem (Algebra.adjoin A (Singleton.singleton ζ) …
          x : B
          hx : Membership.mem (Algebra.adjoin A (Singleton.singleton ζ)) x
          b : B
          hb✝ : Membership.mem (Singleton.singleton ζ) b
          hb : Eq b ζ
          ⊢ Eq (HPow.hPow ⟨ζ, ⋯⟩ ↑n) 1
        -/
        rw [← Subalgebra.coe_eq_one, Subalgebra.coe_pow, Subtype.coe_mk]
        /-
          case refine_1
          A : Type u
          B : Type v
          inst✝² : CommRing A
          inst✝¹ : CommRing B
          inst✝ : Algebra A B
          ζ : B
          n : PNat
          h : IsPrimitiveRoot ζ ↑n
          x✝ : Subtype fun x => Membership.mem (Algebra.adjoin A (Singleton.singleton ζ) …
          x : B
          hx : Membership.mem (Algebra.adjoin A (Singleton.singleton ζ)) x
          b : B
          hb✝ : Membership.mem (Singleton.singleton ζ) b
          hb : Eq b ζ
          ⊢ Eq (HPow.hPow ζ ↑n) 1
        -/
        exact ((IsPrimitiveRoot.iff_def ζ n).1 h).1
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          A : Type u
          B : Type v
          inst✝² : CommRing A
          inst✝¹ : CommRing B
          inst✝ : Algebra A B
          ζ : B
          n : PNat
          h : IsPrimitiveRoot ζ ↑n
          x✝ : Subtype fun x => Membership.mem (Algebra.adjoin A (Singleton.singleton ζ) …
          x : B
          hx : Membership.mem (Algebra.adjoin A (Singleton.singleton ζ)) x
          a : A
          ⊢ Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (Memb …
        -/
      · exact Subalgebra.algebraMap_mem _ _
        /-
          🎉 no goals
        -/
        /-
          case refine_3
          A : Type u
          B : Type v
          inst✝² : CommRing A
          inst✝¹ : CommRing B
          inst✝ : Algebra A B
          ζ : B
          n : PNat
          h : IsPrimitiveRoot ζ ↑n
          x✝² : Subtype fun x => Membership.mem (Algebra.adjoin A (Singleton.singleton ζ …
          x : B
          hx : Membership.mem (Algebra.adjoin A (Singleton.singleton ζ)) x
          b₁ b₂ : B
          x✝¹ : Membership.mem (Algebra.adjoin A (Singleton.singleton ζ)) b₁
          x✝ : Membership.mem (Algebra.adjoin A (Singleton.singleton ζ)) b₂
          hb₁ : Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And ( …
          hb₂ : Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And ( …
          ⊢ Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (Memb …
        -/
      · exact Subalgebra.add_mem _ hb₁ hb₂
        /-
          🎉 no goals
        -/
        /-
          case refine_4
          A : Type u
          B : Type v
          inst✝² : CommRing A
          inst✝¹ : CommRing B
          inst✝ : Algebra A B
          ζ : B
          n : PNat
          h : IsPrimitiveRoot ζ ↑n
          x✝² : Subtype fun x => Membership.mem (Algebra.adjoin A (Singleton.singleton ζ …
          x : B
          hx : Membership.mem (Algebra.adjoin A (Singleton.singleton ζ)) x
          b₁ b₂ : B
          x✝¹ : Membership.mem (Algebra.adjoin A (Singleton.singleton ζ)) b₁
          x✝ : Membership.mem (Algebra.adjoin A (Singleton.singleton ζ)) b₂
          hb₁ : Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And ( …
          hb₂ : Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And ( …
          ⊢ Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (Memb …
        -/
      · exact Subalgebra.mul_mem _ hb₁ hb₂ }
        /-
          🎉 no goals
        -/


/-- A cyclotomic extension splits `X ^ n - 1` if `n ∈ S`. -/
theorem splits_X_pow_sub_one [H : IsCyclotomicExtension S K L] (hS : n ∈ S) :
    Splits (algebraMap K L) (X ^ (n : ℕ) - 1) := by
  rw [← splits_id_iff_splits, Polynomial.map_sub, Polynomial.map_one, Polynomial.map_pow,
    Polynomial.map_X]
  /-
    n : PNat
    S : Set PNat
    K : Type w
    L : Type z
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    H : IsCyclotomicExtension S K L
    hS : Membership.mem S n
    ⊢ Polynomial.Splits (RingHom.id L) (HSub.hSub (HPow.hPow Polynomial.X ↑n) 1)
  -/
  obtain ⟨z, hz⟩ := ((isCyclotomicExtension_iff _ _ _).1 H).1 hS
  /-
    case intro
    n : PNat
    S : Set PNat
    K : Type w
    L : Type z
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    H : IsCyclotomicExtension S K L
    hS : Membership.mem S n
    z : L
    hz : IsPrimitiveRoot z ↑n
    ⊢ Polynomial.Splits (RingHom.id L) (HSub.hSub (HPow.hPow Polynomial.X ↑n) 1)
  -/
  exact X_pow_sub_one_splits hz
  /-
    🎉 no goals
  -/


/-- A cyclotomic extension splits `cyclotomic n K` if `n ∈ S`. -/
theorem splits_cyclotomic [IsCyclotomicExtension S K L] (hS : n ∈ S) :
    Splits (algebraMap K L) (cyclotomic n K) := by
  /-
    n : PNat
    S : Set PNat
    K : Type w
    L : Type z
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension S K L
    hS : Membership.mem S n
    ⊢ Polynomial.Splits (algebraMap K L) (Polynomial.cyclotomic (↑n) K)
  -/
  refine splits_of_splits_of_dvd _ (X_pow_sub_C_ne_zero n.pos _) (splits_X_pow_sub_one K L hS) ?_
  /-
    n : PNat
    S : Set PNat
    K : Type w
    L : Type z
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension S K L
    hS : Membership.mem S n
    ⊢ Dvd.dvd (Polynomial.cyclotomic (↑n) K) (HSub.hSub (HPow.hPow Polynomial.X ↑n …
  -/
  use ∏ i ∈ (n : ℕ).properDivisors, Polynomial.cyclotomic i K
  /-
    case h
    n : PNat
    S : Set PNat
    K : Type w
    L : Type z
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : IsCyclotomicExtension S K L
    hS : Membership.mem S n
    ⊢ Eq (HSub.hSub (HPow.hPow Polynomial.X ↑n) (Polynomial.C 1)) (HMul.hMul (Poly …
  -/
  rw [(eq_cyclotomic_iff n.pos _).1 rfl, RingHom.map_one]
  /-
    🎉 no goals
  -/


/-- If `IsCyclotomicExtension {n} K L`, then `L` is the splitting field of `X ^ n - 1`. -/
theorem isSplittingField_X_pow_sub_one : IsSplittingField K L (X ^ (n : ℕ) - 1) :=
  { splits' := splits_X_pow_sub_one K L (mem_singleton n)
    adjoin_rootSet' := by
      /-
        n : PNat
        K : Type w
        L : Type z
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        ⊢ Eq (Algebra.adjoin K ((HSub.hSub (HPow.hPow Polynomial.X ↑n) 1).rootSet L))  …
      -/
      rw [← ((iff_adjoin_eq_top {n} K L).1 inferInstance).2]
      /-
        n : PNat
        K : Type w
        L : Type z
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        ⊢ Eq (Algebra.adjoin K ((HSub.hSub (HPow.hPow Polynomial.X ↑n) 1).rootSet L))  …
      -/
      congr
      /-
        case e_s
        n : PNat
        K : Type w
        L : Type z
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        ⊢ Eq ((HSub.hSub (HPow.hPow Polynomial.X ↑n) 1).rootSet L) (setOf fun b => Exi …
      -/
      refine Set.ext fun x => ?_
      simp only [Polynomial.map_pow, mem_singleton_iff, Multiset.mem_toFinset, exists_eq_left,
        mem_setOf_eq, Polynomial.map_X, Polynomial.map_one, Finset.mem_coe, Polynomial.map_sub]
      simp only [mem_rootSet', map_sub, map_pow, aeval_one, aeval_X, sub_eq_zero, map_X,
        and_iff_right_iff_imp, Polynomial.map_sub, Polynomial.map_pow, Polynomial.map_one]
      /-
        case e_s
        n : PNat
        K : Type w
        L : Type z
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        x : L
        ⊢ Eq (HPow.hPow x ↑n) 1 → Ne (HSub.hSub (HPow.hPow Polynomial.X ↑n) 1) 0
      -/
      exact fun _ => X_pow_sub_C_ne_zero n.pos (1 : L) }
      /-
        🎉 no goals
      -/


/-- Any two `n`-th cyclotomic extensions are isomorphic. -/
def algEquiv (L' : Type*) [Field L'] [Algebra K L'] [IsCyclotomicExtension {n} K L'] :
    L ≃ₐ[K] L' :=
  let h₁ := isSplittingField_X_pow_sub_one n K L
  let h₂ := isSplittingField_X_pow_sub_one n K L'
  (@IsSplittingField.algEquiv K L _ _ _ (X ^ (n : ℕ) - 1) h₁).trans
    (@IsSplittingField.algEquiv K L' _ _ _ (X ^ (n : ℕ) - 1) h₂).symm


include n in
theorem isGalois : IsGalois K L :=
  letI := isSplittingField_X_pow_sub_one n K L
  IsGalois.of_separable_splitting_field (X_pow_sub_one_separable_iff.2
    (IsCyclotomicExtension.neZero' n K L).1)


/-- If `IsCyclotomicExtension {n} K L`, then `L` is the splitting field of `cyclotomic n K`. -/
theorem splitting_field_cyclotomic : IsSplittingField K L (cyclotomic n K) :=
  { splits' := splits_cyclotomic K L (mem_singleton n)
    adjoin_rootSet' := by
      /-
        n : PNat
        K : Type w
        L : Type z
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        ⊢ Eq (Algebra.adjoin K ((Polynomial.cyclotomic (↑n) K).rootSet L)) Top.top
      -/
      rw [← ((iff_adjoin_eq_top {n} K L).1 inferInstance).2]
      /-
        n : PNat
        K : Type w
        L : Type z
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        ⊢ Eq (Algebra.adjoin K ((Polynomial.cyclotomic (↑n) K).rootSet L)) (Algebra.ad …
      -/
      letI := Classical.decEq L
      -- todo: make `exists_prim_root` take an explicit `L`
      /-
        n : PNat
        K : Type w
        L : Type z
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        this : DecidableEq L := Classical.decEq L
        ⊢ Eq (Algebra.adjoin K ((Polynomial.cyclotomic (↑n) K).rootSet L)) (Algebra.ad …
      -/
      obtain ⟨ζ : L, hζ⟩ := IsCyclotomicExtension.exists_prim_root K (B := L) (mem_singleton n)
      /-
        case intro
        n : PNat
        K : Type w
        L : Type z
        inst✝³ : Field K
        inst✝² : Field L
        inst✝¹ : Algebra K L
        inst✝ : IsCyclotomicExtension (Singleton.singleton n) K L
        this : DecidableEq L := Classical.decEq L
        ζ : L
        hζ : IsPrimitiveRoot ζ ↑n
        ⊢ Eq (Algebra.adjoin K ((Polynomial.cyclotomic (↑n) K).rootSet L)) (Algebra.ad …
      -/
      exact adjoin_roots_cyclotomic_eq_adjoin_nth_roots hζ }
      /-
        🎉 no goals
      -/


/-- Given `n : ℕ+` and a field `K`, we define `CyclotomicField n K` as the
splitting field of `cyclotomic n K`. If `n` is nonzero in `K`, it has
the instance `IsCyclotomicExtension {n} K (CyclotomicField n K)`. -/
def CyclotomicField : Type w :=
  (cyclotomic n K).SplittingField


instance : Field (CyclotomicField n K) := by
  /-
    n : PNat
    S T : Set PNat
    A : Type u
    B : Type v
    K : Type w
    L : Type z
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra A B
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ⊢ Field (CyclotomicField n K)
  -/
  delta CyclotomicField; infer_instance
                         /-
                           🎉 no goals
                         -/

-- Porting note: could not be derived

instance algebra : Algebra K (CyclotomicField n K) := by
  /-
    n : PNat
    S T : Set PNat
    A : Type u
    B : Type v
    K : Type w
    L : Type z
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra A B
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ⊢ Algebra K (CyclotomicField n K)
  -/
  delta CyclotomicField; infer_instance
                         /-
                           🎉 no goals
                         -/

-- Porting note: could not be derived

instance : Inhabited (CyclotomicField n K) := by
  /-
    n : PNat
    S T : Set PNat
    A : Type u
    B : Type v
    K : Type w
    L : Type z
    inst✝⁵ : CommRing A
    inst✝⁴ : CommRing B
    inst✝³ : Algebra A B
    inst✝² : Field K
    inst✝¹ : Field L
    inst✝ : Algebra K L
    ⊢ Inhabited (CyclotomicField n K)
  -/
  delta CyclotomicField; infer_instance
                         /-
                           🎉 no goals
                         -/


instance [CharZero K] : CharZero (CyclotomicField n K) :=
  charZero_of_injective_algebraMap (algebraMap K _).injective


instance isCyclotomicExtension [NeZero ((n : ℕ) : K)] :
    IsCyclotomicExtension {n} K (CyclotomicField n K) := by
  haveI : NeZero ((n : ℕ) : CyclotomicField n K) :=
    NeZero.nat_of_injective (algebraMap K _).injective
  /-
    n : PNat
    S T : Set PNat
    A : Type u
    B : Type v
    K : Type w
    L : Type z
    inst✝⁶ : CommRing A
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra A B
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : NeZero ↑↑n
    this : NeZero ↑↑n
    ⊢ IsCyclotomicExtension (Singleton.singleton n) K (CyclotomicField n K)
  -/
  letI := Classical.decEq (CyclotomicField n K)
  obtain ⟨ζ, hζ⟩ :=
    exists_root_of_splits (algebraMap K (CyclotomicField n K)) (SplittingField.splits _)
      (degree_cyclotomic_pos n K n.pos).ne'
  /-
    case intro
    n : PNat
    S T : Set PNat
    A : Type u
    B : Type v
    K : Type w
    L : Type z
    inst✝⁶ : CommRing A
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra A B
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : NeZero ↑↑n
    this✝ : NeZero ↑↑n
    this : DecidableEq (CyclotomicField n K) := Classical.decEq (CyclotomicField n …
    ζ : CyclotomicField n K
    hζ : Eq (Polynomial.eval₂ (algebraMap K (CyclotomicField n K)) ζ (Polynomial.c …
    ⊢ IsCyclotomicExtension (Singleton.singleton n) K (CyclotomicField n K)
  -/
  rw [← eval_map, ← IsRoot.def, map_cyclotomic, isRoot_cyclotomic_iff] at hζ
-- Porting note: the first `?_` was `forall_eq.2 ⟨ζ, hζ⟩` that now fails.
  /-
    case intro
    n : PNat
    S T : Set PNat
    A : Type u
    B : Type v
    K : Type w
    L : Type z
    inst✝⁶ : CommRing A
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra A B
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : NeZero ↑↑n
    this✝ : NeZero ↑↑n
    this : DecidableEq (CyclotomicField n K) := Classical.decEq (CyclotomicField n …
    ζ : CyclotomicField n K
    hζ : IsPrimitiveRoot ζ ↑n
    ⊢ IsCyclotomicExtension (Singleton.singleton n) K (CyclotomicField n K)
  -/
  refine ⟨?_, ?_⟩
    /-
      case intro.refine_1
      n : PNat
      S T : Set PNat
      A : Type u
      B : Type v
      K : Type w
      L : Type z
      inst✝⁶ : CommRing A
      inst✝⁵ : CommRing B
      inst✝⁴ : Algebra A B
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : NeZero ↑↑n
      this✝ : NeZero ↑↑n
      this : DecidableEq (CyclotomicField n K) := Classical.decEq (CyclotomicField n …
      ζ : CyclotomicField n K
      hζ : IsPrimitiveRoot ζ ↑n
      ⊢ ∀ {n_1 : PNat}, Membership.mem (Singleton.singleton n) n_1 → Exists fun r => …
    -/
  · simp only [mem_singleton_iff, forall_eq]
    /-
      case intro.refine_1
      n : PNat
      S T : Set PNat
      A : Type u
      B : Type v
      K : Type w
      L : Type z
      inst✝⁶ : CommRing A
      inst✝⁵ : CommRing B
      inst✝⁴ : Algebra A B
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : NeZero ↑↑n
      this✝ : NeZero ↑↑n
      this : DecidableEq (CyclotomicField n K) := Classical.decEq (CyclotomicField n …
      ζ : CyclotomicField n K
      hζ : IsPrimitiveRoot ζ ↑n
      ⊢ Exists fun r => IsPrimitiveRoot r ↑n
    -/
    exact ⟨ζ, hζ⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      n : PNat
      S T : Set PNat
      A : Type u
      B : Type v
      K : Type w
      L : Type z
      inst✝⁶ : CommRing A
      inst✝⁵ : CommRing B
      inst✝⁴ : Algebra A B
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : NeZero ↑↑n
      this✝ : NeZero ↑↑n
      this : DecidableEq (CyclotomicField n K) := Classical.decEq (CyclotomicField n …
      ζ : CyclotomicField n K
      hζ : IsPrimitiveRoot ζ ↑n
      ⊢ ∀ (x : CyclotomicField n K), Membership.mem (Algebra.adjoin K (setOf fun b = …
    -/
  · rw [← Algebra.eq_top_iff, ← SplittingField.adjoin_rootSet, eq_comm]
    /-
      case intro.refine_2
      n : PNat
      S T : Set PNat
      A : Type u
      B : Type v
      K : Type w
      L : Type z
      inst✝⁶ : CommRing A
      inst✝⁵ : CommRing B
      inst✝⁴ : Algebra A B
      inst✝³ : Field K
      inst✝² : Field L
      inst✝¹ : Algebra K L
      inst✝ : NeZero ↑↑n
      this✝ : NeZero ↑↑n
      this : DecidableEq (CyclotomicField n K) := Classical.decEq (CyclotomicField n …
      ζ : CyclotomicField n K
      hζ : IsPrimitiveRoot ζ ↑n
      ⊢ Eq (Algebra.adjoin K ((Polynomial.cyclotomic (↑n) K).rootSet (Polynomial.cyc …
    -/
    exact IsCyclotomicExtension.adjoin_roots_cyclotomic_eq_adjoin_nth_roots hζ
    /-
      🎉 no goals
    -/


/-- If `K` is the fraction field of `A`, the `A`-algebra structure on `CyclotomicField n K`.
-/
@[nolint unusedArguments]
instance CyclotomicField.algebraBase : Algebra A (CyclotomicField n K) :=
  SplittingField.algebra' (cyclotomic n K)


instance CyclotomicField.algebra' {R : Type*} [CommRing R] [Algebra R K] :
    Algebra R (CyclotomicField n K) :=
  SplittingField.algebra' (cyclotomic n K)


instance {R : Type*} [CommRing R] [Algebra R K] : IsScalarTower R K (CyclotomicField n K) :=
  SplittingField.isScalarTower _


instance CyclotomicField.noZeroSMulDivisors [IsFractionRing A K] :
    NoZeroSMulDivisors A (CyclotomicField n K) := by
  /-
    n : PNat
    S T : Set PNat
    A : Type u
    B : Type v
    K : Type w
    L : Type z
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra A B
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    ⊢ NoZeroSMulDivisors A (CyclotomicField n K)
  -/
  refine NoZeroSMulDivisors.of_algebraMap_injective ?_
  /-
    n : PNat
    S T : Set PNat
    A : Type u
    B : Type v
    K : Type w
    L : Type z
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra A B
    inst✝⁴ : Field K
    inst✝³ : Field L
    inst✝² : Algebra K L
    inst✝¹ : Algebra A K
    inst✝ : IsFractionRing A K
    ⊢ Function.Injective ⇑(algebraMap A (CyclotomicField n K))
  -/
  rw [IsScalarTower.algebraMap_eq A K (CyclotomicField n K)]
  exact
    (Function.Injective.comp (NoZeroSMulDivisors.algebraMap_injective K (CyclotomicField n K))
      (IsFractionRing.injective A K) : _)


/-- If `A` is a domain with fraction field `K` and `n : ℕ+`, we define `CyclotomicRing n A K` as
the `A`-subalgebra of `CyclotomicField n K` generated by the roots of `X ^ n - 1`. If `n`
is nonzero in `A`, it has the instance `IsCyclotomicExtension {n} A (CyclotomicRing n A K)`. -/
@[nolint unusedArguments]
def CyclotomicRing : Type w :=
  adjoin A {b : CyclotomicField n K | b ^ (n : ℕ) = 1}
--deriving CommRing, IsDomain, Inhabited


instance : CommRing (CyclotomicRing n A K) := by
  /-
    n : PNat
    S T : Set PNat
    A : Type u
    B : Type v
    K : Type w
    L : Type z
    inst✝⁶ : CommRing A
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra A B
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Algebra A K
    ⊢ CommRing (CyclotomicRing n A K)
  -/
  delta CyclotomicRing; infer_instance
                        /-
                          🎉 no goals
                        -/

-- Porting note: could not be derived

instance : IsDomain (CyclotomicRing n A K) := by
  /-
    n : PNat
    S T : Set PNat
    A : Type u
    B : Type v
    K : Type w
    L : Type z
    inst✝⁶ : CommRing A
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra A B
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Algebra A K
    ⊢ IsDomain (CyclotomicRing n A K)
  -/
  delta CyclotomicRing; infer_instance
                        /-
                          🎉 no goals
                        -/

-- Porting note: could not be derived

instance : Inhabited (CyclotomicRing n A K) := by
  /-
    n : PNat
    S T : Set PNat
    A : Type u
    B : Type v
    K : Type w
    L : Type z
    inst✝⁶ : CommRing A
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra A B
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    inst✝ : Algebra A K
    ⊢ Inhabited (CyclotomicRing n A K)
  -/
  delta CyclotomicRing; infer_instance
                        /-
                          🎉 no goals
                        -/


/-- The `A`-algebra structure on `CyclotomicRing n A K`. -/
instance algebraBase : Algebra A (CyclotomicRing n A K) :=
  (adjoin A _).algebra

-- Ensure that there is no diamonds with ℤ.
-- but there is at `reducible_and_instances` https://github.com/leanprover-community/mathlib4/issues/10906

instance [IsFractionRing A K] :
    NoZeroSMulDivisors A (CyclotomicRing n A K) :=
  (adjoin A _).noZeroSMulDivisors_bot


theorem algebraBase_injective [IsFractionRing A K] :
    Function.Injective <| algebraMap A (CyclotomicRing n A K) :=
  NoZeroSMulDivisors.algebraMap_injective _ _


instance : Algebra (CyclotomicRing n A K) (CyclotomicField n K) :=
  (adjoin A _).toAlgebra


theorem adjoin_algebra_injective :
    Function.Injective <| algebraMap (CyclotomicRing n A K) (CyclotomicField n K) :=
  Subtype.val_injective


instance : NoZeroSMulDivisors (CyclotomicRing n A K) (CyclotomicField n K) :=
  NoZeroSMulDivisors.of_algebraMap_injective (adjoin_algebra_injective n A K)


instance : IsScalarTower A (CyclotomicRing n A K) (CyclotomicField n K) :=
  IsScalarTower.subalgebra' _ _ _ _


instance isCyclotomicExtension [IsFractionRing A K] [NeZero ((n : ℕ) : A)] :
    IsCyclotomicExtension {n} A (CyclotomicRing n A K) where
  exists_prim_root := @fun a han => by
    /-
      n : PNat
      S T : Set PNat
      A : Type u
      B : Type v
      K : Type w
      L : Type z
      inst✝⁸ : CommRing A
      inst✝⁷ : CommRing B
      inst✝⁶ : Algebra A B
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : NeZero ↑↑n
      a : PNat
      han : Membership.mem (Singleton.singleton n) a
      ⊢ Exists fun r => IsPrimitiveRoot r ↑a
    -/
    rw [mem_singleton_iff] at han
    /-
      n : PNat
      S T : Set PNat
      A : Type u
      B : Type v
      K : Type w
      L : Type z
      inst✝⁸ : CommRing A
      inst✝⁷ : CommRing B
      inst✝⁶ : Algebra A B
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : NeZero ↑↑n
      a : PNat
      han : Eq a n
      ⊢ Exists fun r => IsPrimitiveRoot r ↑a
    -/
    subst a
    /-
      n : PNat
      S T : Set PNat
      A : Type u
      B : Type v
      K : Type w
      L : Type z
      inst✝⁸ : CommRing A
      inst✝⁷ : CommRing B
      inst✝⁶ : Algebra A B
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : NeZero ↑↑n
      ⊢ Exists fun r => IsPrimitiveRoot r ↑n
    -/
    haveI := NeZero.of_noZeroSMulDivisors A K n
    /-
      n : PNat
      S T : Set PNat
      A : Type u
      B : Type v
      K : Type w
      L : Type z
      inst✝⁸ : CommRing A
      inst✝⁷ : CommRing B
      inst✝⁶ : Algebra A B
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : NeZero ↑↑n
      this : NeZero ↑↑n
      ⊢ Exists fun r => IsPrimitiveRoot r ↑n
    -/
    haveI := NeZero.of_noZeroSMulDivisors A (CyclotomicField n K) n
    /-
      n : PNat
      S T : Set PNat
      A : Type u
      B : Type v
      K : Type w
      L : Type z
      inst✝⁸ : CommRing A
      inst✝⁷ : CommRing B
      inst✝⁶ : Algebra A B
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : NeZero ↑↑n
      this✝ : NeZero ↑↑n
      this : NeZero ↑↑n
      ⊢ Exists fun r => IsPrimitiveRoot r ↑n
    -/
    obtain ⟨μ, hμ⟩ := (CyclotomicField.isCyclotomicExtension n K).exists_prim_root (mem_singleton n)
    /-
      case intro
      n : PNat
      S T : Set PNat
      A : Type u
      B : Type v
      K : Type w
      L : Type z
      inst✝⁸ : CommRing A
      inst✝⁷ : CommRing B
      inst✝⁶ : Algebra A B
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : NeZero ↑↑n
      this✝ : NeZero ↑↑n
      this : NeZero ↑↑n
      μ : CyclotomicField n K
      hμ : IsPrimitiveRoot μ ↑n
      ⊢ Exists fun r => IsPrimitiveRoot r ↑n
    -/
    refine ⟨⟨μ, subset_adjoin ?_⟩, ?_⟩
      /-
        case intro.refine_1
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁸ : CommRing A
        inst✝⁷ : CommRing B
        inst✝⁶ : Algebra A B
        inst✝⁵ : Field K
        inst✝⁴ : Field L
        inst✝³ : Algebra K L
        inst✝² : Algebra A K
        inst✝¹ : IsFractionRing A K
        inst✝ : NeZero ↑↑n
        this✝ : NeZero ↑↑n
        this : NeZero ↑↑n
        μ : CyclotomicField n K
        hμ : IsPrimitiveRoot μ ↑n
        ⊢ Membership.mem (setOf fun b => Eq (HPow.hPow b ↑n) 1) μ
      -/
    · apply (isRoot_of_unity_iff n.pos (CyclotomicField n K)).mpr
      /-
        case intro.refine_1
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁸ : CommRing A
        inst✝⁷ : CommRing B
        inst✝⁶ : Algebra A B
        inst✝⁵ : Field K
        inst✝⁴ : Field L
        inst✝³ : Algebra K L
        inst✝² : Algebra A K
        inst✝¹ : IsFractionRing A K
        inst✝ : NeZero ↑↑n
        this✝ : NeZero ↑↑n
        this : NeZero ↑↑n
        μ : CyclotomicField n K
        hμ : IsPrimitiveRoot μ ↑n
        ⊢ Exists fun i => And (Membership.mem (↑n).divisors i) ((Polynomial.cyclotomic …
      -/
      refine ⟨n, Nat.mem_divisors_self _ n.ne_zero, ?_⟩
      /-
        case intro.refine_1
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁸ : CommRing A
        inst✝⁷ : CommRing B
        inst✝⁶ : Algebra A B
        inst✝⁵ : Field K
        inst✝⁴ : Field L
        inst✝³ : Algebra K L
        inst✝² : Algebra A K
        inst✝¹ : IsFractionRing A K
        inst✝ : NeZero ↑↑n
        this✝ : NeZero ↑↑n
        this : NeZero ↑↑n
        μ : CyclotomicField n K
        hμ : IsPrimitiveRoot μ ↑n
        ⊢ (Polynomial.cyclotomic (↑n) (CyclotomicField n K)).IsRoot μ
      -/
      rwa [← isRoot_cyclotomic_iff] at hμ
      /-
        🎉 no goals
      -/
      /-
        case intro.refine_2
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁸ : CommRing A
        inst✝⁷ : CommRing B
        inst✝⁶ : Algebra A B
        inst✝⁵ : Field K
        inst✝⁴ : Field L
        inst✝³ : Algebra K L
        inst✝² : Algebra A K
        inst✝¹ : IsFractionRing A K
        inst✝ : NeZero ↑↑n
        this✝ : NeZero ↑↑n
        this : NeZero ↑↑n
        μ : CyclotomicField n K
        hμ : IsPrimitiveRoot μ ↑n
        ⊢ IsPrimitiveRoot ⟨μ, ⋯⟩ ↑n
      -/
    · rwa [← IsPrimitiveRoot.coe_submonoidClass_iff, Subtype.coe_mk]
      /-
        🎉 no goals
      -/
  adjoin_roots x := by
    /-
      n : PNat
      S T : Set PNat
      A : Type u
      B : Type v
      K : Type w
      L : Type z
      inst✝⁸ : CommRing A
      inst✝⁷ : CommRing B
      inst✝⁶ : Algebra A B
      inst✝⁵ : Field K
      inst✝⁴ : Field L
      inst✝³ : Algebra K L
      inst✝² : Algebra A K
      inst✝¹ : IsFractionRing A K
      inst✝ : NeZero ↑↑n
      x : CyclotomicRing n A K
      ⊢ Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (Memb …
    -/
    obtain ⟨x, hx⟩ := x
    refine
      adjoin_induction (fun y hy => ?_) (fun a => ?_) (fun y z _ _ hy hz => ?_)
        (fun y z  _ _ hy hz => ?_) hx
      /-
        case mk.refine_1
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁸ : CommRing A
        inst✝⁷ : CommRing B
        inst✝⁶ : Algebra A B
        inst✝⁵ : Field K
        inst✝⁴ : Field L
        inst✝³ : Algebra K L
        inst✝² : Algebra A K
        inst✝¹ : IsFractionRing A K
        inst✝ : NeZero ↑↑n
        x : CyclotomicField n K
        hx : Membership.mem (Algebra.adjoin A (setOf fun b => Eq (HPow.hPow b ↑n) 1)) x
        y : CyclotomicField n K
        hy : Membership.mem (setOf fun b => Eq (HPow.hPow b ↑n) 1) y
        ⊢ Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (Memb …
      -/
    · refine subset_adjoin ?_
      /-
        case mk.refine_1
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁸ : CommRing A
        inst✝⁷ : CommRing B
        inst✝⁶ : Algebra A B
        inst✝⁵ : Field K
        inst✝⁴ : Field L
        inst✝³ : Algebra K L
        inst✝² : Algebra A K
        inst✝¹ : IsFractionRing A K
        inst✝ : NeZero ↑↑n
        x : CyclotomicField n K
        hx : Membership.mem (Algebra.adjoin A (setOf fun b => Eq (HPow.hPow b ↑n) 1)) x
        y : CyclotomicField n K
        hy : Membership.mem (setOf fun b => Eq (HPow.hPow b ↑n) 1) y
        ⊢ Membership.mem (setOf fun b => Exists fun n_1 => And (Membership.mem (Single …
      -/
      simp only [mem_singleton_iff, exists_eq_left, mem_setOf_eq]
      /-
        case mk.refine_1
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁸ : CommRing A
        inst✝⁷ : CommRing B
        inst✝⁶ : Algebra A B
        inst✝⁵ : Field K
        inst✝⁴ : Field L
        inst✝³ : Algebra K L
        inst✝² : Algebra A K
        inst✝¹ : IsFractionRing A K
        inst✝ : NeZero ↑↑n
        x : CyclotomicField n K
        hx : Membership.mem (Algebra.adjoin A (setOf fun b => Eq (HPow.hPow b ↑n) 1)) x
        y : CyclotomicField n K
        hy : Membership.mem (setOf fun b => Eq (HPow.hPow b ↑n) 1) y
        ⊢ Eq (HPow.hPow ⟨y, ⋯⟩ ↑n) 1
      -/
      rwa [← Subalgebra.coe_eq_one, Subalgebra.coe_pow, Subtype.coe_mk]
      /-
        🎉 no goals
      -/
      /-
        case mk.refine_2
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁸ : CommRing A
        inst✝⁷ : CommRing B
        inst✝⁶ : Algebra A B
        inst✝⁵ : Field K
        inst✝⁴ : Field L
        inst✝³ : Algebra K L
        inst✝² : Algebra A K
        inst✝¹ : IsFractionRing A K
        inst✝ : NeZero ↑↑n
        x : CyclotomicField n K
        hx : Membership.mem (Algebra.adjoin A (setOf fun b => Eq (HPow.hPow b ↑n) 1)) x
        a : A
        ⊢ Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (Memb …
      -/
    · exact Subalgebra.algebraMap_mem _ a
      /-
        🎉 no goals
      -/
      /-
        case mk.refine_3
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁸ : CommRing A
        inst✝⁷ : CommRing B
        inst✝⁶ : Algebra A B
        inst✝⁵ : Field K
        inst✝⁴ : Field L
        inst✝³ : Algebra K L
        inst✝² : Algebra A K
        inst✝¹ : IsFractionRing A K
        inst✝ : NeZero ↑↑n
        x : CyclotomicField n K
        hx : Membership.mem (Algebra.adjoin A (setOf fun b => Eq (HPow.hPow b ↑n) 1)) x
        y z : CyclotomicField n K
        x✝¹ : Membership.mem (Algebra.adjoin A (setOf fun b => Eq (HPow.hPow b ↑n) 1)) y
        x✝ : Membership.mem (Algebra.adjoin A (setOf fun b => Eq (HPow.hPow b ↑n) 1)) z
        hy : Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (M …
        hz : Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (M …
        ⊢ Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (Memb …
      -/
    · exact Subalgebra.add_mem _ hy hz
      /-
        🎉 no goals
      -/
      /-
        case mk.refine_4
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁸ : CommRing A
        inst✝⁷ : CommRing B
        inst✝⁶ : Algebra A B
        inst✝⁵ : Field K
        inst✝⁴ : Field L
        inst✝³ : Algebra K L
        inst✝² : Algebra A K
        inst✝¹ : IsFractionRing A K
        inst✝ : NeZero ↑↑n
        x : CyclotomicField n K
        hx : Membership.mem (Algebra.adjoin A (setOf fun b => Eq (HPow.hPow b ↑n) 1)) x
        y z : CyclotomicField n K
        x✝¹ : Membership.mem (Algebra.adjoin A (setOf fun b => Eq (HPow.hPow b ↑n) 1)) y
        x✝ : Membership.mem (Algebra.adjoin A (setOf fun b => Eq (HPow.hPow b ↑n) 1)) z
        hy : Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (M …
        hz : Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (M …
        ⊢ Membership.mem (Algebra.adjoin A (setOf fun b => Exists fun n_1 => And (Memb …
      -/
    · exact Subalgebra.mul_mem _ hy hz
      /-
        🎉 no goals
      -/


instance [IsFractionRing A K] [IsDomain A] [NeZero ((n : ℕ) : A)] :
    IsFractionRing (CyclotomicRing n A K) (CyclotomicField n K) where
  map_units' := fun ⟨x, hx⟩ => by
    /-
      n : PNat
      S T : Set PNat
      A : Type u
      B : Type v
      K : Type w
      L : Type z
      inst✝⁹ : CommRing A
      inst✝⁸ : CommRing B
      inst✝⁷ : Algebra A B
      inst✝⁶ : Field K
      inst✝⁵ : Field L
      inst✝⁴ : Algebra K L
      inst✝³ : Algebra A K
      inst✝² : IsFractionRing A K
      inst✝¹ : IsDomain A
      inst✝ : NeZero ↑↑n
      x✝ : Subtype fun x => Membership.mem (nonZeroDivisors (CyclotomicRing n A K)) x
      x : CyclotomicRing n A K
      hx : Membership.mem (nonZeroDivisors (CyclotomicRing n A K)) x
      ⊢ IsUnit ((algebraMap (CyclotomicRing n A K) (CyclotomicField n K)) ↑⟨x, hx⟩)
    -/
    rw [isUnit_iff_ne_zero]
    /-
      n : PNat
      S T : Set PNat
      A : Type u
      B : Type v
      K : Type w
      L : Type z
      inst✝⁹ : CommRing A
      inst✝⁸ : CommRing B
      inst✝⁷ : Algebra A B
      inst✝⁶ : Field K
      inst✝⁵ : Field L
      inst✝⁴ : Algebra K L
      inst✝³ : Algebra A K
      inst✝² : IsFractionRing A K
      inst✝¹ : IsDomain A
      inst✝ : NeZero ↑↑n
      x✝ : Subtype fun x => Membership.mem (nonZeroDivisors (CyclotomicRing n A K)) x
      x : CyclotomicRing n A K
      hx : Membership.mem (nonZeroDivisors (CyclotomicRing n A K)) x
      ⊢ Ne ((algebraMap (CyclotomicRing n A K) (CyclotomicField n K)) ↑⟨x, hx⟩) 0
    -/
    apply map_ne_zero_of_mem_nonZeroDivisors
      /-
        case hg
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁹ : CommRing A
        inst✝⁸ : CommRing B
        inst✝⁷ : Algebra A B
        inst✝⁶ : Field K
        inst✝⁵ : Field L
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra A K
        inst✝² : IsFractionRing A K
        inst✝¹ : IsDomain A
        inst✝ : NeZero ↑↑n
        x✝ : Subtype fun x => Membership.mem (nonZeroDivisors (CyclotomicRing n A K)) x
        x : CyclotomicRing n A K
        hx : Membership.mem (nonZeroDivisors (CyclotomicRing n A K)) x
        ⊢ Function.Injective ⇑(algebraMap (CyclotomicRing n A K) (CyclotomicField n K))
      -/
    · apply adjoin_algebra_injective
      /-
        🎉 no goals
      -/
      /-
        case h
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁹ : CommRing A
        inst✝⁸ : CommRing B
        inst✝⁷ : Algebra A B
        inst✝⁶ : Field K
        inst✝⁵ : Field L
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra A K
        inst✝² : IsFractionRing A K
        inst✝¹ : IsDomain A
        inst✝ : NeZero ↑↑n
        x✝ : Subtype fun x => Membership.mem (nonZeroDivisors (CyclotomicRing n A K)) x
        x : CyclotomicRing n A K
        hx : Membership.mem (nonZeroDivisors (CyclotomicRing n A K)) x
        ⊢ Membership.mem (nonZeroDivisors (CyclotomicRing n A K)) ↑⟨x, hx⟩
      -/
    · exact hx
      /-
        🎉 no goals
      -/
  surj' x := by
    /-
      n : PNat
      S T : Set PNat
      A : Type u
      B : Type v
      K : Type w
      L : Type z
      inst✝⁹ : CommRing A
      inst✝⁸ : CommRing B
      inst✝⁷ : Algebra A B
      inst✝⁶ : Field K
      inst✝⁵ : Field L
      inst✝⁴ : Algebra K L
      inst✝³ : Algebra A K
      inst✝² : IsFractionRing A K
      inst✝¹ : IsDomain A
      inst✝ : NeZero ↑↑n
      x : CyclotomicField n K
      ⊢ Exists fun x_1 => Eq (HMul.hMul x ((algebraMap (CyclotomicRing n A K) (Cyclo …
    -/
    letI : NeZero ((n : ℕ) : K) := NeZero.nat_of_injective (IsFractionRing.injective A K)
    refine
      Algebra.adjoin_induction
        (hx := ((IsCyclotomicExtension.iff_singleton n K (CyclotomicField n K)).1
            (CyclotomicField.isCyclotomicExtension n K)).2 x)
        (fun y hy => ?_) (fun k => ?_) ?_ ?_
-- Porting note: the last goal was `by simpa` that now fails.
      /-
        case refine_1
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁹ : CommRing A
        inst✝⁸ : CommRing B
        inst✝⁷ : Algebra A B
        inst✝⁶ : Field K
        inst✝⁵ : Field L
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra A K
        inst✝² : IsFractionRing A K
        inst✝¹ : IsDomain A
        inst✝ : NeZero ↑↑n
        x : CyclotomicField n K
        this : NeZero ↑↑n := NeZero.nat_of_injective (IsFractionRing.injective A K)
        y : CyclotomicField n K
        hy : Membership.mem (setOf fun b => Eq (HPow.hPow b ↑n) 1) y
        ⊢ Exists fun x => Eq (HMul.hMul y ((algebraMap (CyclotomicRing n A K) (Cycloto …
      -/
    · exact ⟨⟨⟨y, subset_adjoin hy⟩, 1⟩, by simp; rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁹ : CommRing A
        inst✝⁸ : CommRing B
        inst✝⁷ : Algebra A B
        inst✝⁶ : Field K
        inst✝⁵ : Field L
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra A K
        inst✝² : IsFractionRing A K
        inst✝¹ : IsDomain A
        inst✝ : NeZero ↑↑n
        x : CyclotomicField n K
        this : NeZero ↑↑n := NeZero.nat_of_injective (IsFractionRing.injective A K)
        k : K
        ⊢ Exists fun x => Eq (HMul.hMul ((algebraMap K (CyclotomicField n K)) k) ((alg …
      -/
    · have : IsLocalization (nonZeroDivisors A) K := inferInstance
      /-
        case refine_2
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁹ : CommRing A
        inst✝⁸ : CommRing B
        inst✝⁷ : Algebra A B
        inst✝⁶ : Field K
        inst✝⁵ : Field L
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra A K
        inst✝² : IsFractionRing A K
        inst✝¹ : IsDomain A
        inst✝ : NeZero ↑↑n
        x : CyclotomicField n K
        this✝ : NeZero ↑↑n := NeZero.nat_of_injective (IsFractionRing.injective A K)
        k : K
        this : IsLocalization (nonZeroDivisors A) K
        ⊢ Exists fun x => Eq (HMul.hMul ((algebraMap K (CyclotomicField n K)) k) ((alg …
      -/
      replace := this.surj
      /-
        case refine_2
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁹ : CommRing A
        inst✝⁸ : CommRing B
        inst✝⁷ : Algebra A B
        inst✝⁶ : Field K
        inst✝⁵ : Field L
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra A K
        inst✝² : IsFractionRing A K
        inst✝¹ : IsDomain A
        inst✝ : NeZero ↑↑n
        x : CyclotomicField n K
        this✝ : NeZero ↑↑n := NeZero.nat_of_injective (IsFractionRing.injective A K)
        k : K
        this : ∀ (z : K), Exists fun x => Eq (HMul.hMul z ((algebraMap A K) ↑x.2)) ((a …
        ⊢ Exists fun x => Eq (HMul.hMul ((algebraMap K (CyclotomicField n K)) k) ((alg …
      -/
      obtain ⟨⟨z, w⟩, hw⟩ := this k
      refine ⟨⟨algebraMap A (CyclotomicRing n A K) z, algebraMap A (CyclotomicRing n A K) w,
        map_mem_nonZeroDivisors _ (algebraBase_injective n A K) w.2⟩, ?_⟩
      letI : IsScalarTower A K (CyclotomicField n K) :=
        IsScalarTower.of_algebraMap_eq (congr_fun rfl)
      rw [← IsScalarTower.algebraMap_apply, ← IsScalarTower.algebraMap_apply,
        @IsScalarTower.algebraMap_apply A K _ _ _ _ _ (_root_.CyclotomicField.algebra n K) _ _ w,
        ← RingHom.map_mul, hw, ← IsScalarTower.algebraMap_apply]
      /-
        case refine_3
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁹ : CommRing A
        inst✝⁸ : CommRing B
        inst✝⁷ : Algebra A B
        inst✝⁶ : Field K
        inst✝⁵ : Field L
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra A K
        inst✝² : IsFractionRing A K
        inst✝¹ : IsDomain A
        inst✝ : NeZero ↑↑n
        x : CyclotomicField n K
        this : NeZero ↑↑n := NeZero.nat_of_injective (IsFractionRing.injective A K)
        ⊢ ∀ (x y : CyclotomicField n K), Membership.mem (Algebra.adjoin K (setOf fun b …
      -/
    · rintro y z - - ⟨a, ha⟩ ⟨b, hb⟩
      /-
        case refine_3.intro.intro
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁹ : CommRing A
        inst✝⁸ : CommRing B
        inst✝⁷ : Algebra A B
        inst✝⁶ : Field K
        inst✝⁵ : Field L
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra A K
        inst✝² : IsFractionRing A K
        inst✝¹ : IsDomain A
        inst✝ : NeZero ↑↑n
        x : CyclotomicField n K
        this : NeZero ↑↑n := NeZero.nat_of_injective (IsFractionRing.injective A K)
        y z : CyclotomicField n K
        a : Prod (CyclotomicRing n A K) (Subtype fun x => Membership.mem (nonZeroDivis …
        ha : Eq (HMul.hMul y ((algebraMap (CyclotomicRing n A K) (CyclotomicField n K) …
        b : Prod (CyclotomicRing n A K) (Subtype fun x => Membership.mem (nonZeroDivis …
        hb : Eq (HMul.hMul z ((algebraMap (CyclotomicRing n A K) (CyclotomicField n K) …
        ⊢ Exists fun x => Eq (HMul.hMul (HAdd.hAdd y z) ((algebraMap (CyclotomicRing n …
      -/
      refine ⟨⟨a.1 * b.2 + b.1 * a.2, a.2 * b.2, mul_mem_nonZeroDivisors.2 ⟨a.2.2, b.2.2⟩⟩, ?_⟩
      rw [RingHom.map_mul, add_mul, ← mul_assoc, ha,
        mul_comm ((algebraMap (CyclotomicRing n A K) _) ↑a.2), ← mul_assoc, hb]
      /-
        case refine_3.intro.intro
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁹ : CommRing A
        inst✝⁸ : CommRing B
        inst✝⁷ : Algebra A B
        inst✝⁶ : Field K
        inst✝⁵ : Field L
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra A K
        inst✝² : IsFractionRing A K
        inst✝¹ : IsDomain A
        inst✝ : NeZero ↑↑n
        x : CyclotomicField n K
        this : NeZero ↑↑n := NeZero.nat_of_injective (IsFractionRing.injective A K)
        y z : CyclotomicField n K
        a : Prod (CyclotomicRing n A K) (Subtype fun x => Membership.mem (nonZeroDivis …
        ha : Eq (HMul.hMul y ((algebraMap (CyclotomicRing n A K) (CyclotomicField n K) …
        b : Prod (CyclotomicRing n A K) (Subtype fun x => Membership.mem (nonZeroDivis …
        hb : Eq (HMul.hMul z ((algebraMap (CyclotomicRing n A K) (CyclotomicField n K) …
        ⊢ Eq (HAdd.hAdd (HMul.hMul ((algebraMap (CyclotomicRing n A K) (CyclotomicFiel …
      -/
      simp only [map_add, map_mul]
      /-
        🎉 no goals
      -/
      /-
        case refine_4
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁹ : CommRing A
        inst✝⁸ : CommRing B
        inst✝⁷ : Algebra A B
        inst✝⁶ : Field K
        inst✝⁵ : Field L
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra A K
        inst✝² : IsFractionRing A K
        inst✝¹ : IsDomain A
        inst✝ : NeZero ↑↑n
        x : CyclotomicField n K
        this : NeZero ↑↑n := NeZero.nat_of_injective (IsFractionRing.injective A K)
        ⊢ ∀ (x y : CyclotomicField n K), Membership.mem (Algebra.adjoin K (setOf fun b …
      -/
    · rintro y z - - ⟨a, ha⟩ ⟨b, hb⟩
      /-
        case refine_4.intro.intro
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁹ : CommRing A
        inst✝⁸ : CommRing B
        inst✝⁷ : Algebra A B
        inst✝⁶ : Field K
        inst✝⁵ : Field L
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra A K
        inst✝² : IsFractionRing A K
        inst✝¹ : IsDomain A
        inst✝ : NeZero ↑↑n
        x : CyclotomicField n K
        this : NeZero ↑↑n := NeZero.nat_of_injective (IsFractionRing.injective A K)
        y z : CyclotomicField n K
        a : Prod (CyclotomicRing n A K) (Subtype fun x => Membership.mem (nonZeroDivis …
        ha : Eq (HMul.hMul y ((algebraMap (CyclotomicRing n A K) (CyclotomicField n K) …
        b : Prod (CyclotomicRing n A K) (Subtype fun x => Membership.mem (nonZeroDivis …
        hb : Eq (HMul.hMul z ((algebraMap (CyclotomicRing n A K) (CyclotomicField n K) …
        ⊢ Exists fun x => Eq (HMul.hMul (HMul.hMul y z) ((algebraMap (CyclotomicRing n …
      -/
      refine ⟨⟨a.1 * b.1, a.2 * b.2, mul_mem_nonZeroDivisors.2 ⟨a.2.2, b.2.2⟩⟩, ?_⟩
      rw [RingHom.map_mul, mul_comm ((algebraMap (CyclotomicRing n A K) _) ↑a.2), mul_assoc, ←
        mul_assoc z, hb, ← mul_comm ((algebraMap (CyclotomicRing n A K) _) ↑a.2), ← mul_assoc, ha]
      /-
        case refine_4.intro.intro
        n : PNat
        S T : Set PNat
        A : Type u
        B : Type v
        K : Type w
        L : Type z
        inst✝⁹ : CommRing A
        inst✝⁸ : CommRing B
        inst✝⁷ : Algebra A B
        inst✝⁶ : Field K
        inst✝⁵ : Field L
        inst✝⁴ : Algebra K L
        inst✝³ : Algebra A K
        inst✝² : IsFractionRing A K
        inst✝¹ : IsDomain A
        inst✝ : NeZero ↑↑n
        x : CyclotomicField n K
        this : NeZero ↑↑n := NeZero.nat_of_injective (IsFractionRing.injective A K)
        y z : CyclotomicField n K
        a : Prod (CyclotomicRing n A K) (Subtype fun x => Membership.mem (nonZeroDivis …
        ha : Eq (HMul.hMul y ((algebraMap (CyclotomicRing n A K) (CyclotomicField n K) …
        b : Prod (CyclotomicRing n A K) (Subtype fun x => Membership.mem (nonZeroDivis …
        hb : Eq (HMul.hMul z ((algebraMap (CyclotomicRing n A K) (CyclotomicField n K) …
        ⊢ Eq (HMul.hMul ((algebraMap (CyclotomicRing n A K) (CyclotomicField n K)) a.1 …
      -/
      simp only [map_mul]
      /-
        🎉 no goals
      -/
                                 /-
                                   n : PNat
                                   S T : Set PNat
                                   A : Type u
                                   B : Type v
                                   K : Type w
                                   L : Type z
                                   inst✝⁹ : CommRing A
                                   inst✝⁸ : CommRing B
                                   inst✝⁷ : Algebra A B
                                   inst✝⁶ : Field K
                                   inst✝⁵ : Field L
                                   inst✝⁴ : Algebra K L
                                   inst✝³ : Algebra A K
                                   inst✝² : IsFractionRing A K
                                   inst✝¹ : IsDomain A
                                   inst✝ : NeZero ↑↑n
                                   x y : CyclotomicRing n A K
                                   h : Eq ((algebraMap (CyclotomicRing n A K) (CyclotomicField n K)) x) ((algebra …
                                   ⊢ Eq (HMul.hMul (↑1) x) (HMul.hMul (↑1) y)
                                 -/
  exists_of_eq {x y} h := ⟨1, by rw [adjoin_algebra_injective n A K h]⟩
                                 /-
                                   🎉 no goals
                                 -/


theorem eq_adjoin_primitive_root {μ : CyclotomicField n K} (h : IsPrimitiveRoot μ n) :
    CyclotomicRing n A K = adjoin A ({μ} : Set (CyclotomicField n K)) := by
  rw [← IsCyclotomicExtension.adjoin_roots_cyclotomic_eq_adjoin_root_cyclotomic h,
    IsCyclotomicExtension.adjoin_roots_cyclotomic_eq_adjoin_nth_roots h]
  /-
    n : PNat
    A : Type u
    K : Type w
    inst✝² : CommRing A
    inst✝¹ : Field K
    inst✝ : Algebra A K
    μ : CyclotomicField n K
    h : IsPrimitiveRoot μ ↑n
    ⊢ Eq (CyclotomicRing n A K) (Subtype fun x => Membership.mem (Algebra.adjoin A …
  -/
  simp [CyclotomicRing]
  /-
    🎉 no goals
  -/


/-- Algebraically closed fields are `S`-cyclotomic extensions over themselves if
`NeZero ((a : ℕ) : K))` for all `a ∈ S`. -/
theorem IsAlgClosed.isCyclotomicExtension (h : ∀ a ∈ S, NeZero ((a : ℕ) : K)) :
    IsCyclotomicExtension S K K := by
  /-
    S : Set PNat
    K : Type w
    inst✝¹ : Field K
    inst✝ : IsAlgClosed K
    h : ∀ (a : PNat), Membership.mem S a → NeZero ↑↑a
    ⊢ IsCyclotomicExtension S K K
  -/
  refine ⟨@fun a ha => ?_, Algebra.eq_top_iff.mp <| Subsingleton.elim _ _⟩
  /-
    S : Set PNat
    K : Type w
    inst✝¹ : Field K
    inst✝ : IsAlgClosed K
    h : ∀ (a : PNat), Membership.mem S a → NeZero ↑↑a
    a : PNat
    ha : Membership.mem S a
    ⊢ Exists fun r => IsPrimitiveRoot r ↑a
  -/
  obtain ⟨r, hr⟩ := IsAlgClosed.exists_aeval_eq_zero K _ (degree_cyclotomic_pos a K a.pos).ne'
  /-
    case intro
    S : Set PNat
    K : Type w
    inst✝¹ : Field K
    inst✝ : IsAlgClosed K
    h : ∀ (a : PNat), Membership.mem S a → NeZero ↑↑a
    a : PNat
    ha : Membership.mem S a
    r : K
    hr : Eq ((Polynomial.aeval r) (Polynomial.cyclotomic (↑a) K)) 0
    ⊢ Exists fun r => IsPrimitiveRoot r ↑a
  -/
  refine ⟨r, ?_⟩
  /-
    case intro
    S : Set PNat
    K : Type w
    inst✝¹ : Field K
    inst✝ : IsAlgClosed K
    h : ∀ (a : PNat), Membership.mem S a → NeZero ↑↑a
    a : PNat
    ha : Membership.mem S a
    r : K
    hr : Eq ((Polynomial.aeval r) (Polynomial.cyclotomic (↑a) K)) 0
    ⊢ IsPrimitiveRoot r ↑a
  -/
  haveI := h a ha
  /-
    case intro
    S : Set PNat
    K : Type w
    inst✝¹ : Field K
    inst✝ : IsAlgClosed K
    h : ∀ (a : PNat), Membership.mem S a → NeZero ↑↑a
    a : PNat
    ha : Membership.mem S a
    r : K
    hr : Eq ((Polynomial.aeval r) (Polynomial.cyclotomic (↑a) K)) 0
    this : NeZero ↑↑a
    ⊢ IsPrimitiveRoot r ↑a
  -/
  rwa [coe_aeval_eq_eval, ← IsRoot.def, isRoot_cyclotomic_iff] at hr
  /-
    🎉 no goals
  -/


instance IsAlgClosedOfCharZero.isCyclotomicExtension [CharZero K] :
    ∀ S, IsCyclotomicExtension S K K := fun S =>
  IsAlgClosed.isCyclotomicExtension S K fun _ _ => inferInstance


